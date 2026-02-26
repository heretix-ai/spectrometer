from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from spectrometer.models import AnonymousUsage, UsageLedger, User

logger = logging.getLogger(__name__)


@dataclass
class UsagePlan:
    name: str
    checks_allowed: int


ANON_PLAN = UsagePlan("anon", checks_allowed=5)
TRIAL_PLAN = UsagePlan("trial", checks_allowed=3)
STARTER_PLAN = UsagePlan("starter", checks_allowed=20)
CORE_PLAN = UsagePlan("core", checks_allowed=100)
PRO_PLAN = UsagePlan("pro", checks_allowed=750)


PLAN_MAP = {
    "anon": ANON_PLAN,
    "trial": TRIAL_PLAN,
    "starter": STARTER_PLAN,
    "core": CORE_PLAN,
    "pro": PRO_PLAN,
}


@dataclass
class UsageState:
    plan: UsagePlan
    checks_used: int
    checks_allowed: int
    remaining: int
    ledger: Optional[UsageLedger] = None
    anon_token: Optional[str] = None
    anon_usage: Optional[AnonymousUsage] = None

    @property
    def enough_credit(self) -> bool:
        return self.remaining > 0


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _current_period(today: date) -> tuple[date, date]:
    start = today.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    end = end.replace(day=1) - date.resolution
    return start, end


def _resolve_plan(user: Optional[User]) -> UsagePlan:
    if not user or not user.plan:
        return TRIAL_PLAN
    return PLAN_MAP.get(user.plan, TRIAL_PLAN)


def get_usage_state(
    session: Session,
    user: Optional[User],
    *,
    anon_token: Optional[str] = None,
) -> UsageState:
    plan = _resolve_plan(user)
    today = _today()

    if not user:
        if not anon_token:
            return UsageState(
                plan=ANON_PLAN,
                checks_used=0,
                checks_allowed=ANON_PLAN.checks_allowed,
                remaining=ANON_PLAN.checks_allowed,
            )
        anon_usage = session.get(AnonymousUsage, anon_token)
        if not anon_usage:
            anon_usage = AnonymousUsage(
                token=anon_token,
                checks_allowed=ANON_PLAN.checks_allowed,
                checks_used=0,
            )
            session.add(anon_usage)
            session.flush()
        used = anon_usage.checks_used
        remaining = max(anon_usage.checks_allowed - used, 0)
        return UsageState(
            plan=ANON_PLAN,
            checks_used=used,
            checks_allowed=anon_usage.checks_allowed,
            remaining=remaining,
            anon_token=anon_token,
            anon_usage=anon_usage,
        )

    period_start, period_end = _current_period(today)
    stmt = select(UsageLedger).where(
        UsageLedger.user_id == user.id,
        UsageLedger.period_start == period_start,
    )
    ledger = session.scalar(stmt)
    if not ledger:
        ledger = UsageLedger(
            user_id=user.id,
            period_start=period_start,
            period_end=period_end,
            plan=plan.name,
            checks_allowed=plan.checks_allowed,
            checks_used=0,
        )
        session.add(ledger)
        session.flush()
    used = ledger.checks_used
    remaining = max(ledger.checks_allowed - used, 0)
    return UsageState(plan=plan, checks_used=used, checks_allowed=ledger.checks_allowed, remaining=remaining, ledger=ledger)


def increment_usage(session: Session, user: Optional[User], state: UsageState) -> int:
    """Increment usage and return total used after the increment."""
    if not user:
        if state.anon_usage is not None and state.anon_usage.checks_used < state.anon_usage.checks_allowed:
            state.anon_usage.checks_used += 1
            session.add(state.anon_usage)
            return state.anon_usage.checks_used
        return min(state.checks_used + 1, state.checks_allowed)
    if not state.ledger:
        logger.warning("No usage ledger for user %s", user.id)
        return state.checks_used
    if state.ledger.checks_used < state.ledger.checks_allowed:
        state.ledger.checks_used += 1
    session.add(state.ledger)
    return state.ledger.checks_used
