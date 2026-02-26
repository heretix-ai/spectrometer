from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import stripe
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from spectrometer.models import User

from .config import settings
from .usage import PLAN_MAP, TRIAL_PLAN, get_usage_state

logger = logging.getLogger(__name__)


def _stripe_client() -> stripe:
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Stripe integration not configured")
    stripe.api_key = settings.stripe_secret_key
    return stripe


def _ensure_customer(session: Session, user: User) -> str:
    client = _stripe_client()
    if user.stripe_customer_id:
        return user.stripe_customer_id
    customer = client.Customer.create(email=user.email)
    user.stripe_customer_id = customer["id"]
    session.add(user)
    return user.stripe_customer_id


def create_checkout_session(session: Session, user: User, plan: str) -> str:
    price_id = settings.price_for_plan(plan)
    if not price_id:
        raise HTTPException(status_code=400, detail="Unsupported plan tier")

    client = _stripe_client()
    customer_id = _ensure_customer(session, user)
    checkout = client.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        allow_promotion_codes=True,
        metadata={"plan": plan},
        subscription_data={"metadata": {"plan": plan}},
        success_url=settings.stripe_success_url(),
        cancel_url=settings.stripe_cancel_url(),
        client_reference_id=str(user.id),
    )
    user.stripe_customer_id = customer_id
    session.add(user)
    return checkout["url"]


def create_portal_session(session: Session, user: User) -> str:
    config_id = settings.stripe_portal_config
    if not config_id:
        raise HTTPException(status_code=503, detail="Stripe customer portal not configured")

    client = _stripe_client()
    customer_id = _ensure_customer(session, user)
    portal_session = client.billing_portal.Session.create(
        customer=customer_id,
        configuration=config_id,
        return_url=settings.stripe_portal_return_url(),
    )
    logger.info("Created portal session for user %s", user.id)
    return portal_session["url"]


def _reset_usage_for_plan(session: Session, user: User, plan: str) -> None:
    state = get_usage_state(session, user)
    if state.ledger:
        state.ledger.checks_allowed = PLAN_MAP[plan].checks_allowed
        state.ledger.checks_used = 0
        state.ledger.plan = plan
        session.add(state.ledger)


def handle_checkout_completed(session: Session, payload: dict) -> None:
    customer_id = payload.get("customer")
    plan = payload.get("metadata", {}).get("plan") or payload.get("subscription_metadata", {}).get("plan")
    subscription_id = payload.get("subscription")
    if not customer_id:
        logger.warning("checkout.session.completed without customer id")
        return
    stmt = select(User).where(User.stripe_customer_id == customer_id)
    user = session.scalar(stmt)
    if not user:
        logger.warning("No user with customer id %s", customer_id)
        return
    if plan not in PLAN_MAP:
        logger.warning("Unknown plan %s", plan)
        return
    user.plan = plan
    if subscription_id:
        user.stripe_subscription_id = subscription_id
    period_start = payload.get("subscription_details", {}).get("current_period_start") or payload.get("current_period_start")
    if isinstance(period_start, int):
        user.billing_anchor = datetime.fromtimestamp(period_start, tz=timezone.utc).date()
    session.add(user)
    _reset_usage_for_plan(session, user, plan)


def handle_subscription_updated(session: Session, payload: dict) -> None:
    subscription_id = payload.get("id")
    if not subscription_id:
        return
    stmt = select(User).where(User.stripe_subscription_id == subscription_id)
    user = session.scalar(stmt)
    if not user:
        return
    plan = payload.get("metadata", {}).get("plan")
    if plan not in PLAN_MAP:
        item = payload.get("items", {}).get("data", [])
        if item:
            price_id = item[0].get("price", {}).get("id")
            for name in PLAN_MAP:
                price_attr = settings.price_for_plan(name)
                if price_attr == price_id:
                    plan = name
                    break
    if plan and plan in PLAN_MAP:
        user.plan = plan
        session.add(user)
        _reset_usage_for_plan(session, user, plan)


def handle_subscription_deleted(session: Session, payload: dict) -> None:
    subscription_id = payload.get("id")
    if not subscription_id:
        return
    stmt = select(User).where(User.stripe_subscription_id == subscription_id)
    user = session.scalar(stmt)
    if not user:
        return
    user.plan = TRIAL_PLAN.name
    user.stripe_subscription_id = None
    session.add(user)
    _reset_usage_for_plan(session, user, user.plan)
