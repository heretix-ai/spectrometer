from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from pydantic import EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from spectrometer.models import EmailToken, Session as DbSession, User

from .config import settings
from .database import get_session
from .email import email_sender

MAGIC_LINK_SEPARATOR = ":"


def _find_user(session: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email)
    return session.scalar(stmt)


def _create_user(session: Session, email: str) -> User:
    user = User(email=email)
    session.add(user)
    session.flush()
    return user


def _hash_verifier(verifier: str) -> str:
    return hashlib.sha256(verifier.encode("utf-8")).hexdigest()


def issue_magic_link(session: Session, email: str) -> None:
    normalized = email.lower().strip()
    user = _find_user(session, normalized)
    if not user:
        user = _create_user(session, normalized)

    now = datetime.now(timezone.utc)
    selector = secrets.token_hex(16)
    verifier = secrets.token_urlsafe(32)
    digest = _hash_verifier(verifier)
    expires_at = now + timedelta(minutes=settings.magic_link_ttl_minutes)

    token = EmailToken(
        user_id=user.id,
        selector=selector,
        verifier_hash=digest,
        created_at=now,
        expires_at=expires_at,
        consumed_at=None,
    )
    session.add(token)

    link = f"{settings.api_url.rstrip('/')}/api/auth/callback?token={selector}{MAGIC_LINK_SEPARATOR}{verifier}"
    email_sender.send_magic_link(normalized, link)


def _consume_magic_token(session: Session, selector: str, verifier: str) -> User:
    stmt = select(EmailToken).where(EmailToken.selector == selector)
    token = session.scalar(stmt)
    if not token:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    if token.consumed_at is not None:
        raise HTTPException(status_code=400, detail="Token already used")
    now = datetime.now(timezone.utc)
    if token.expires_at < now:
        raise HTTPException(status_code=400, detail="Token expired")
    if token.verifier_hash != _hash_verifier(verifier):
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    token.consumed_at = now
    session.add(token)
    user = session.get(User, token.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    return user


def _create_session(session: Session, user: User) -> DbSession:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=settings.session_ttl_days)
    session_id = uuid.uuid4()
    s = DbSession(
        id=session_id,
        user_id=user.id,
        created_at=now,
        last_seen_at=now,
        expires_at=expires_at,
        user_agent=None,
    )
    session.add(s)
    return s


def handle_magic_link(email: EmailStr, session: Session) -> None:
    issue_magic_link(session, email)


def complete_magic_link(token_param: str, session: Session) -> JSONResponse:
    if not token_param or MAGIC_LINK_SEPARATOR not in token_param:
        raise HTTPException(status_code=400, detail="Invalid token")
    selector, verifier = token_param.split(MAGIC_LINK_SEPARATOR, 1)
    user = _consume_magic_token(session, selector, verifier)
    db_session = _create_session(session, user)

    redirect_target = settings.app_url.rstrip("/") + "/?signed=1"
    response = RedirectResponse(url=redirect_target, status_code=303)

    cookie_params = {
        "key": settings.session_cookie_name,
        "value": str(db_session.id),
        "httponly": True,
        "max_age": settings.session_ttl_days * 86400,
        "samesite": "lax",
        "path": "/",
    }
    if settings.session_cookie_secure:
        cookie_params["secure"] = True
    if settings.session_cookie_domain:
        cookie_params["domain"] = settings.session_cookie_domain
    response.set_cookie(**cookie_params)
    return response


def sign_out(request: Request, session: Session) -> Response:
    token = request.cookies.get(settings.session_cookie_name)
    if token:
        try:
            session_id = uuid.UUID(token)
        except ValueError:
            session_id = None
        if session_id:
            db_session = session.get(DbSession, session_id)
            if db_session:
                session.delete(db_session)

    response = Response(status_code=204)
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        domain=settings.session_cookie_domain or None,
    )
    return response


def get_current_user(request: Request, session: Session = Depends(get_session)) -> Optional[User]:
    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None
    try:
        session_id = uuid.UUID(token)
    except (ValueError, TypeError):
        return None
    db_session = session.get(DbSession, session_id)
    if not db_session:
        return None
    now = datetime.now(timezone.utc)
    if db_session.expires_at < now:
        session.delete(db_session)
        return None
    db_session.last_seen_at = now
    session.add(db_session)
    user = session.get(User, db_session.user_id)
    return user
