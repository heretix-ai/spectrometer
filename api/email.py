from __future__ import annotations

import logging
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


class EmailSender:
    """Simple email helper supporting Postmark or console fallback."""

    def __init__(self, token: Optional[str], sender: str) -> None:
        self.token = token
        self.sender = sender

    def send_magic_link(self, recipient: str, link: str) -> None:
        subject = "Sign in to Spectrometer"
        text = (
            "Here is your sign-in link:\n\n"
            f"{link}\n\n"
            "This link will expire shortly. If you did not request it, you can ignore this email."
        )
        if not self.token:
            logger.info("Magic link for %s: %s", recipient, link)
            return

        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    "https://api.postmarkapp.com/email",
                    headers={
                        "Accept": "application/json",
                        "X-Postmark-Server-Token": self.token,
                    },
                    json={
                        "From": self.sender,
                        "To": recipient,
                        "Subject": subject,
                        "TextBody": text,
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Postmark send failed (%s): %s", recipient, exc)
            logger.info("Magic link for %s: %s", recipient, link)

    def send_alert(self, recipient: str, subject: str, text: str) -> None:
        if not recipient:
            return
        if not self.token:
            logger.info("Alert email (dry-run) to=%s subject=%s body=%s", recipient, subject, text)
            return
        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                resp = client.post(
                    "https://api.postmarkapp.com/email",
                    headers={
                        "Accept": "application/json",
                        "X-Postmark-Server-Token": self.token,
                    },
                    json={
                        "From": self.sender,
                        "To": recipient,
                        "Subject": subject,
                        "TextBody": text,
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("Postmark alert send failed (%s): %s", recipient, exc)


email_sender = EmailSender(settings.postmark_token, settings.email_sender_address)
