"""
Bias Spectrometer API.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import logging
import secrets
from pathlib import Path
from typing import Optional

import stripe
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from spectrometer.engine import run_spectrometer, result_to_dict
from spectrometer.models import Query, User

from .auth import complete_magic_link, get_current_user, handle_magic_link, sign_out
from .billing import (
    create_checkout_session,
    create_portal_session,
    handle_checkout_completed,
    handle_subscription_deleted,
    handle_subscription_updated,
)
from .config import settings
from .database import get_session
from .usage import UsageState, get_usage_state, increment_usage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent.parent / "ui"

app = FastAPI(title="Bias Spectrometer", docs_url="/docs" if settings.app_env == "local" else None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

MAX_CLAIM_CHARS = 500


# ── Helpers ─────────────────────────────────────────────────────────


def _ensure_anon_token(request: Request, response: Response) -> Optional[str]:
    """Return the anonymous token from cookie, creating one if needed."""
    token = request.cookies.get(settings.anon_cookie_name)
    if not token:
        token = secrets.token_hex(16)
        response.set_cookie(
            key=settings.anon_cookie_name,
            value=token,
            httponly=True,
            max_age=365 * 86400,
            samesite="lax",
            path="/",
        )
    return token


# ── Health ──────────────────────────────────────────────────────────


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ── Spectrometer endpoint ───────────────────────────────────────────


class RunRequest(BaseModel):
    claim: str


class RunResponse(BaseModel):
    claim: str
    consensus: dict
    models: list
    meta: dict
    usage: dict


@app.post("/api/spectrometer")
def api_spectrometer(
    body: RunRequest,
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    claim = body.claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="Missing claim")
    if len(claim) > MAX_CLAIM_CHARS:
        raise HTTPException(status_code=400, detail=f"Claim too long (max {MAX_CLAIM_CHARS} chars)")

    # Usage gating
    anon_token = _ensure_anon_token(request, response)
    state = get_usage_state(session, user, anon_token=anon_token)
    if not state.enough_credit:
        reason = "require_subscription" if user else "require_signin"
        raise HTTPException(status_code=402, detail={"reason": reason, "message": "Usage limit reached"})

    # Run spectrometer
    result = run_spectrometer(claim)
    data = result_to_dict(result)

    # Increment usage
    new_used = increment_usage(session, user, state)

    # Store query history
    query = Query(
        user_id=user.id if user else None,
        anon_token=anon_token if not user else None,
        claim=claim,
        consensus_prob=result.consensus_prob,
        consensus_spread=result.consensus_spread,
        agreement=result.agreement,
        response_json=data,
        latency_ms=result.total_latency_ms,
    )
    session.add(query)

    # Add usage info to response
    data["usage"] = {
        "plan": state.plan.name,
        "checks_used": new_used,
        "checks_allowed": state.checks_allowed,
        "remaining": max(state.checks_allowed - new_used, 0),
    }

    return data


# ── Auth endpoints ──────────────────────────────────────────────────


class MagicLinkRequest(BaseModel):
    email: EmailStr


@app.post("/api/auth/magic-links", status_code=204)
def request_magic_link(body: MagicLinkRequest, session: Session = Depends(get_session)):
    handle_magic_link(body.email, session)


@app.get("/api/auth/callback")
def auth_callback(token: str, session: Session = Depends(get_session)):
    return complete_magic_link(token, session)


@app.post("/api/auth/signout", status_code=204)
def signout(request: Request, session: Session = Depends(get_session)):
    return sign_out(request, session)


# ── User info ───────────────────────────────────────────────────────


@app.get("/api/me")
def me(
    request: Request,
    response: Response,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    anon_token = _ensure_anon_token(request, response)
    state = get_usage_state(session, user, anon_token=anon_token)

    if user:
        return {
            "authenticated": True,
            "email": user.email,
            "plan": user.plan,
            "usage_plan": state.plan.name,
            "checks_allowed": state.checks_allowed,
            "checks_used": state.checks_used,
            "remaining": state.remaining,
        }
    return {
        "authenticated": False,
        "usage_plan": state.plan.name,
        "checks_allowed": state.checks_allowed,
        "checks_used": state.checks_used,
        "remaining": state.remaining,
    }


# ── Billing endpoints ──────────────────────────────────────────────


class CheckoutRequest(BaseModel):
    plan: str


@app.post("/api/billing/checkout")
def billing_checkout(
    body: CheckoutRequest,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    url = create_checkout_session(session, user, body.plan)
    return {"checkout_url": url}


@app.post("/api/billing/portal")
def billing_portal(
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(get_current_user),
):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    url = create_portal_session(session, user)
    return {"portal_url": url}


@app.post("/api/stripe/webhook", status_code=204)
async def stripe_webhook(request: Request, session: Session = Depends(get_session)):
    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")

    body = await request.body()
    sig = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(body, sig, settings.stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    payload = event["data"]["object"]

    if event_type == "checkout.session.completed":
        handle_checkout_completed(session, payload)
    elif event_type == "customer.subscription.updated":
        handle_subscription_updated(session, payload)
    elif event_type in ("customer.subscription.deleted", "customer.subscription.cancelled"):
        handle_subscription_deleted(session, payload)
    else:
        logger.info("Unhandled Stripe event: %s", event_type)


# ── Static UI serving ──────────────────────────────────────────────


@app.get("/")
def serve_index():
    html_path = UI_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h1>Bias Spectrometer</h1><p>UI not found.</p>")


@app.get("/billing/success")
@app.get("/billing/success/")
def billing_success():
    path = UI_DIR / "billing" / "success" / "index.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Payment successful!</h1>")


@app.get("/billing/cancel")
@app.get("/billing/cancel/")
def billing_cancel():
    path = UI_DIR / "billing" / "cancel" / "index.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    return HTMLResponse("<h1>Payment cancelled.</h1>")
