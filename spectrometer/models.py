"""SQLAlchemy ORM models for the Bias Spectrometer."""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for ORM models."""


UUID_TYPE = PG_UUID(as_uuid=True)
JSON_TYPE = JSONB


# ── Auth & billing ───────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    plan: Mapped[str] = mapped_column(String(32), nullable=False, default="trial")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    billing_anchor: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    sessions: Mapped[List["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    email_tokens: Mapped[List["EmailToken"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    usage_periods: Mapped[List["UsageLedger"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    queries: Mapped[List["Query"]] = relationship(back_populates="user")


class Session(Base):
    """Active authenticated session."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    user: Mapped["User"] = relationship(back_populates="sessions")

    __table_args__ = (Index("ix_sessions_user_id", "user_id"),)


class EmailToken(Base):
    __tablename__ = "email_tokens"

    id: Mapped[uuid.UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    selector: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    verifier_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="email_tokens")

    __table_args__ = (Index("ix_email_tokens_user_id", "user_id"),)


# ── Usage tracking ───────────────────────────────────────────────────


class UsageLedger(Base):
    __tablename__ = "usage_ledger"

    id: Mapped[uuid.UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID_TYPE, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    plan: Mapped[str] = mapped_column(String(32), nullable=False)
    checks_allowed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checks_used: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="usage_periods")

    __table_args__ = (
        UniqueConstraint("user_id", "period_start", name="uq_usage_period"),
        Index("ix_usage_user", "user_id"),
    )


class AnonymousUsage(Base):
    __tablename__ = "anonymous_usage"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    checks_allowed: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    checks_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


# ── Spectrometer queries (history) ───────────────────────────────────


class Query(Base):
    """A single spectrometer query with full response."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(UUID_TYPE, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID_TYPE, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    anon_token: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    consensus_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    consensus_spread: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    agreement: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    response_json: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[Optional["User"]] = relationship(back_populates="queries")

    __table_args__ = (
        Index("ix_queries_user_id", "user_id"),
        Index("ix_queries_anon_token", "anon_token"),
        Index("ix_queries_created_at", "created_at"),
    )
