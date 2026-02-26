from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="allow", case_sensitive=False)

    app_env: str = Field("local", alias="APP_ENV")
    database_url: str = Field(
        "postgresql+psycopg://spectrometer:spectrometer@localhost:5433/spectrometer",
        alias="DATABASE_URL",
    )
    api_url: str = Field("http://127.0.0.1:8000", alias="API_URL")
    app_url: str = Field("http://127.0.0.1:8000", alias="APP_URL")

    # Auth
    magic_link_ttl_minutes: int = Field(10, alias="MAGIC_LINK_TTL_MINUTES")
    session_ttl_days: int = Field(30, alias="SESSION_TTL_DAYS")
    session_cookie_name: str = Field("spectrometer_session", alias="SESSION_COOKIE_NAME")
    session_cookie_domain: Optional[str] = Field(None, alias="SESSION_COOKIE_DOMAIN")
    session_cookie_secure: bool = Field(False, alias="SESSION_COOKIE_SECURE")
    anon_cookie_name: str = Field("spectrometer_anon", alias="ANON_COOKIE_NAME")

    # Email
    email_sender_address: str = Field("hello@heretix.ai", alias="EMAIL_SENDER_ADDRESS")
    postmark_token: Optional[str] = Field(None, alias="POSTMARK_TOKEN")

    # Stripe
    stripe_secret_key: Optional[str] = Field(None, alias="STRIPE_SECRET")
    stripe_webhook_secret: Optional[str] = Field(None, alias="STRIPE_WEBHOOK_SECRET")
    stripe_price_starter: Optional[str] = Field(None, alias="STRIPE_PRICE_STARTER")
    stripe_price_core: Optional[str] = Field(None, alias="STRIPE_PRICE_CORE")
    stripe_price_pro: Optional[str] = Field(None, alias="STRIPE_PRICE_PRO")
    stripe_success_path: str = Field("/billing/success", alias="STRIPE_SUCCESS_PATH")
    stripe_cancel_path: str = Field("/billing/cancel", alias="STRIPE_CANCEL_PATH")
    stripe_portal_config: Optional[str] = Field(None, alias="STRIPE_PORTAL_CONFIG")
    stripe_portal_return_path: str = Field("/", alias="STRIPE_PORTAL_RETURN_PATH")

    # API keys
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    gemini_api_key: Optional[str] = Field(None, alias="GEMINI_API_KEY")
    xai_api_key: Optional[str] = Field(None, alias="XAI_API_KEY")

    # Alerts
    alert_email: Optional[str] = Field(None, alias="ALERT_EMAIL")
    alert_cooldown_seconds: int = Field(300, alias="ALERT_COOLDOWN_SECONDS")

    def price_for_plan(self, plan: str) -> Optional[str]:
        mapping = {
            "starter": self.stripe_price_starter,
            "core": self.stripe_price_core,
            "pro": self.stripe_price_pro,
        }
        return mapping.get(plan)

    def stripe_success_url(self) -> str:
        return f"{self.app_url.rstrip('/')}{self.stripe_success_path}"

    def stripe_cancel_url(self) -> str:
        return f"{self.app_url.rstrip('/')}{self.stripe_cancel_path}"

    def stripe_portal_return_url(self) -> str:
        return f"{self.app_url.rstrip('/')}{self.stripe_portal_return_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
