#!/usr/bin/env python3
"""
Standalone server for the Bias Spectrometer.

    uv run python serve.py

Serves the UI and runs spectrometer queries. No database, no auth.
"""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
PORT_DEFAULT = 8000
MAX_CLAIM_CHARS = 500

logging.basicConfig(level=logging.INFO, format="[spec] %(message)s")
logger = logging.getLogger(__name__)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info(fmt, *args)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html_path = ROOT / "ui" / "index.html"
            self._ok(html_path.read_bytes(), "text/html")
        else:
            self._not_found()

    def do_POST(self):
        if self.path != "/api/spectrometer":
            self._json_err("Not found", 404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._json_err("Invalid JSON", 400)
            return

        claim = (payload.get("claim") or "").strip()
        if not claim:
            self._json_err("Missing claim", 400)
            return
        if len(claim) > MAX_CLAIM_CHARS:
            self._json_err(f"Claim too long (max {MAX_CLAIM_CHARS} chars)", 400)
            return

        try:
            from spectrometer.engine import run_spectrometer, result_to_dict

            result = run_spectrometer(claim)
            data = result_to_dict(result)
            self._json_ok(data)
        except Exception as e:
            logger.exception("Spectrometer failed")
            self._json_err(str(e)[:500], 500)

    def _ok(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_ok(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_err(self, msg: str, status: int = 500):
        body = json.dumps({"error": msg}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self):
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    load_dotenv()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", str(PORT_DEFAULT)))

    from spectrometer.engine import available_models
    models = available_models()
    model_names = ", ".join(m.name for m in models) or "NONE (set API keys)"

    httpd = HTTPServer((host, port), Handler)
    print(f"Bias Spectrometer running at http://{host}:{port}")
    print(f"Models: {model_names}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
