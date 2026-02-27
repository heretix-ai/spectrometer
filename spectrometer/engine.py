"""
Bias Spectrometer — single-call weight extraction pipeline.

Fires 2 calls per model x N models in parallel. Returns per-model
probability + certainty (spread) + signal text, plus cross-model consensus.
Total wall clock: ~4 seconds.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a calibrated probability estimator. "
    "Evaluate the following claim and return your honest probability estimate "
    "that it is true, from 0 to 100. "
    "Do not hedge. Do not say \"it depends.\" Commit to a specific number. "
    "Extreme values (below 5 or above 95) are correct when warranted.\n\n"
    "You must also assess your epistemic grounding:\n"
    "- \"specific\": you have direct training data about the exact facts in this claim\n"
    "- \"general\": you are extrapolating from general knowledge, not claim-specific facts\n"
    "- \"none\": you have no relevant training data for this claim\n\n"
    "Respond with JSON only:\n"
    "{"
    '"prob_true": <0-100>, '
    '"signal": "<1-2 sentences on what drives this>", '
    '"grounding": "<specific|general|none>", '
    '"grounding_note": "<1 sentence: what you do/don\'t have training data for>"'
    "}"
)

CALLS_PER_MODEL = int(os.getenv("SPEC_CALLS", "3"))
CALL_TIMEOUT = float(os.getenv("SPEC_TIMEOUT", "25"))

# ── Model registry ───────────────────────────────────────────────────


@dataclass
class ModelDef:
    id: str
    name: str
    provider: str  # openai | gemini | xai
    api_model: str
    color: str  # hex for UI dot


MODELS: list[ModelDef] = [
    ModelDef("gpt5", "GPT-5.2", "openai", "gpt-5.2", "#10b981"),
    ModelDef("gemini", "Gemini 3 Flash", "gemini", "gemini-3-flash-preview", "#6366f1"),
    ModelDef("grok", "Grok 4.1", "xai", "grok-4-1-fast-non-reasoning", "#f59e0b"),
]


def available_models() -> list[ModelDef]:
    """Return models whose API keys are configured."""
    checks = {
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    }
    out = []
    for m in MODELS:
        keys = checks.get(m.provider, ())
        if any(os.getenv(k) for k in keys):
            out.append(m)
    return out


# ── Provider calls ───────────────────────────────────────────────────


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Last resort: extract prob_true from truncated JSON (e.g. Gemini thinking overflow)
    prob_match = re.search(r'"prob_true"\s*:\s*(\d+(?:\.\d+)?)', text)
    if prob_match:
        partial = {"prob_true": float(prob_match.group(1))}
        # Match both closed "..." and unclosed strings truncated at end
        sig_match = re.search(r'"signal"\s*:\s*"([^"]*)"?', text)
        if sig_match:
            partial["signal"] = sig_match.group(1)
        gr_match = re.search(r'"grounding"\s*:\s*"([^"]*)"?', text)
        if gr_match:
            partial["grounding"] = gr_match.group(1)
        gn_match = re.search(r'"grounding_note"\s*:\s*"([^"]*)"?', text)
        if gn_match:
            partial["grounding_note"] = gn_match.group(1)
        return partial
    raise ValueError(f"No JSON found in: {text[:200]}")


def _call_openai(claim: str, model: str) -> dict:
    from openai import OpenAI

    client = OpenAI(timeout=CALL_TIMEOUT, max_retries=0)
    try:
        resp = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f'Claim: "{claim}"'}
                    ],
                }
            ],
            max_output_tokens=256,
        )
        text = resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Claim: "{claim}"'},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
    return _extract_json(text)


def _call_gemini(claim: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": f'Claim: "{claim}"'}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "max_output_tokens": 2048,
        },
    }
    r = requests.post(url, params={"key": api_key}, json=payload, timeout=CALL_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates in Gemini response")
    candidate = candidates[0]
    fr = candidate.get("finishReason", "")
    if fr == "SAFETY":
        raise ValueError("Safety filter blocked this claim")
    if fr in ("RECITATION", "OTHER"):
        raise ValueError(f"Gemini blocked: {fr}")
    parts = candidate.get("content", {}).get("parts", [])
    if not parts:
        if fr == "MAX_TOKENS":
            raise ValueError("Response truncated (output too long)")
        raise ValueError("No content in Gemini response")
    text = parts[0].get("text", "")
    try:
        return _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        if fr == "MAX_TOKENS" or len(text.strip()) < 20:
            raise ValueError("Response truncated — partial JSON returned")
        raise


def _call_xai(claim: str, model: str) -> dict:
    from openai import OpenAI as XAIClient

    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    client = XAIClient(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        timeout=CALL_TIMEOUT,
        max_retries=0,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'Claim: "{claim}"'},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    return _extract_json(text)


_CALLERS = {
    "openai": _call_openai,
    "gemini": _call_gemini,
    "xai": _call_xai,
}


def _extract_prob(raw: dict) -> Optional[float]:
    for key in ("prob_true", "bet_true", "probability"):
        if key in raw:
            val = raw[key]
            if isinstance(val, (int, float)):
                v = float(val)
                # Normalize 0-1 scale to 0-100 (values like 0.05, 0.85)
                # Only apply if strictly between 0 and 1 exclusive —
                # 0 and 1 are ambiguous but values like 0.5 are clearly decimal
                if 0 < v < 1:
                    v = v * 100
                return v
    return None


# ── Result types ─────────────────────────────────────────────────────


@dataclass
class ModelReading:
    model_id: str
    model_name: str
    color: str
    probability: Optional[float]  # 0-100 average
    readings: list[Optional[float]]  # individual call values
    spread: Optional[float]  # max - min of readings
    certainty: str  # "locked" | "stable" | "contested" | "refused"
    signal: str  # explanation from first successful call
    grounding: str  # "specific" | "general" | "none" | "unknown"
    grounding_note: str  # what the model does/doesn't know
    latency_ms: int
    error: Optional[str] = None


@dataclass
class ConsensusResult:
    claim: str
    models: list[ModelReading]
    consensus_prob: Optional[float]  # mean of available model probs
    consensus_spread: Optional[float]  # max - min across models
    agreement: str  # "strong" | "moderate" | "divided"
    summary: str  # one-line consensus text
    n_responded: int
    n_total: int
    total_latency_ms: int


def _certainty_label(spread: Optional[float], n_valid: int = 0, n_total: int = 0) -> str:
    if spread is None:
        return "refused"
    if n_valid < n_total:
        # Some calls failed — downgrade confidence
        if spread <= 10:
            return "partial"
        return "contested"
    if spread <= 3:
        return "locked"
    if spread <= 10:
        return "stable"
    return "contested"


def _agreement_label(spread: Optional[float]) -> str:
    if spread is None:
        return "divided"
    if spread <= 8:
        return "strong"
    if spread <= 20:
        return "moderate"
    return "divided"


def _build_summary(result: ConsensusResult) -> str:
    responded = [m for m in result.models if m.probability is not None]
    refused = [m for m in result.models if m.probability is None]

    if not responded:
        return "All models declined to answer this claim."

    n = len(responded)
    avg = result.consensus_prob

    if avg is None:
        return "Unable to compute consensus."

    if avg >= 80:
        direction = "strongly lean toward true"
    elif avg >= 60:
        direction = "lean toward true"
    elif avg >= 40:
        direction = "are uncertain"
    elif avg >= 20:
        direction = "lean toward false"
    else:
        direction = "strongly lean toward false"

    if result.agreement == "strong":
        consensus_text = f"All {n} models {direction}"
    elif result.agreement == "moderate":
        lo_model = min(responded, key=lambda m: m.probability)
        hi_model = max(responded, key=lambda m: m.probability)
        consensus_text = (
            f"Models mostly {direction} — "
            f"{lo_model.model_name} says {lo_model.probability:.0f}%, "
            f"{hi_model.model_name} says {hi_model.probability:.0f}%"
        )
    else:
        lo_model = min(responded, key=lambda m: m.probability)
        hi_model = max(responded, key=lambda m: m.probability)
        consensus_text = (
            f"Models disagree — "
            f"{lo_model.model_name} says {lo_model.probability:.0f}%, "
            f"{hi_model.model_name} says {hi_model.probability:.0f}%"
        )

    if refused:
        names = ", ".join(m.model_name for m in refused)
        consensus_text += f". {names} declined to answer."

    return consensus_text


# ── Main entry point ─────────────────────────────────────────────────


def _normalize_grounding(raw_val: Any) -> str:
    """Normalize grounding value to one of: specific, general, none, unknown."""
    if not isinstance(raw_val, str):
        return "unknown"
    val = raw_val.strip().lower()
    if val in ("specific", "general", "none"):
        return val
    if "specific" in val:
        return "specific"
    if "general" in val:
        return "general"
    if "none" in val or "no " in val:
        return "none"
    return "unknown"


def _call_single(model: ModelDef, claim: str, call_idx: int) -> dict:
    """Execute one API call with one retry. Returns a dict with prob, signal, grounding, latency, error."""
    caller = _CALLERS[model.provider]
    t0 = time.time()
    last_error = None
    for attempt in range(2):
        try:
            raw = caller(claim, model.api_model)
            prob = _extract_prob(raw)
            if attempt > 0:
                logger.info("%s call %d recovered on retry", model.id, call_idx)
            return {
                "prob": prob,
                "signal": raw.get("signal", ""),
                "grounding": _normalize_grounding(raw.get("grounding")),
                "grounding_note": str(raw.get("grounding_note") or ""),
                "latency_ms": int((time.time() - t0) * 1000),
                "error": None,
            }
        except Exception as e:
            last_error = e
            logger.warning("%s call %d attempt %d failed: %s", model.id, call_idx, attempt + 1, str(e)[:200])
            if attempt == 0:
                time.sleep(0.5)
    return {
        "prob": None,
        "signal": "",
        "grounding": "unknown",
        "grounding_note": "",
        "latency_ms": int((time.time() - t0) * 1000),
        "error": str(last_error)[:300] if last_error else "Unknown error",
        }


def run_spectrometer(
    claim: str,
    *,
    models: Optional[list[ModelDef]] = None,
    calls_per_model: int = CALLS_PER_MODEL,
) -> ConsensusResult:
    """Run the bias spectrometer on a claim. Returns ConsensusResult."""
    if models is None:
        models = available_models()
    if not models:
        raise RuntimeError("No models are configured (check API keys)")

    t0 = time.time()

    # Fire all calls in parallel
    futures: dict[Any, tuple[ModelDef, int]] = {}
    max_workers = len(models) * calls_per_model
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for model in models:
            for i in range(calls_per_model):
                f = pool.submit(_call_single, model, claim, i)
                futures[f] = (model, i)

    # Collect results grouped by model
    raw_results: dict[str, list[dict]] = {m.id: [] for m in models}
    for f in as_completed(futures):
        model, idx = futures[f]
        raw_results[model.id].append(f.result())

    # Build ModelReading for each model
    model_readings: list[ModelReading] = []
    for model in models:
        calls = raw_results[model.id]
        probs = [c["prob"] for c in calls if c["prob"] is not None]
        signals = [c["signal"] for c in calls if c["signal"]]
        groundings = [c["grounding"] for c in calls if c["grounding"] != "unknown"]
        grounding_notes = [c["grounding_note"] for c in calls if c["grounding_note"]]
        errors = [c["error"] for c in calls if c["error"]]
        latencies = [c["latency_ms"] for c in calls]

        if probs:
            avg_prob = mean(probs)
            spread = max(probs) - min(probs) if len(probs) > 1 else 0.0
            certainty = _certainty_label(spread, n_valid=len(probs), n_total=len(calls))
        else:
            avg_prob = None
            spread = None
            certainty = "refused"

        grounding = groundings[0] if groundings else "unknown"
        grounding_note = grounding_notes[0] if grounding_notes else ""

        # Pick best signal — skip useless short ones like "False" or "True"
        signal = ""
        for s in signals:
            if len(s) > 10:
                signal = s
                break
        # Fall back to grounding_note if signal is still empty/useless
        if not signal and grounding_note and len(grounding_note) > 10:
            signal = grounding_note

        reading = ModelReading(
            model_id=model.id,
            model_name=model.name,
            color=model.color,
            probability=round(avg_prob, 1) if avg_prob is not None else None,
            readings=[c["prob"] for c in calls],
            spread=round(spread, 1) if spread is not None else None,
            certainty=certainty,
            signal=signal,
            grounding=grounding,
            grounding_note=grounding_note,
            latency_ms=max(latencies) if latencies else 0,
            error=errors[0] if errors and not probs else None,
        )
        model_readings.append(reading)

    # Cross-model consensus
    responding = [m for m in model_readings if m.probability is not None]
    if responding:
        all_probs = [m.probability for m in responding]
        consensus_prob = round(mean(all_probs), 1)
        consensus_spread = round(max(all_probs) - min(all_probs), 1)
        agreement = _agreement_label(consensus_spread)
    else:
        consensus_prob = None
        consensus_spread = None
        agreement = "divided"

    total_ms = int((time.time() - t0) * 1000)

    result = ConsensusResult(
        claim=claim,
        models=model_readings,
        consensus_prob=consensus_prob,
        consensus_spread=consensus_spread,
        agreement=agreement,
        summary="",
        n_responded=len(responding),
        n_total=len(models),
        total_latency_ms=total_ms,
    )
    result.summary = _build_summary(result)
    return result


def result_to_dict(result: ConsensusResult) -> dict:
    """Convert ConsensusResult to a JSON-serializable dict."""
    return {
        "claim": result.claim,
        "consensus": {
            "probability": result.consensus_prob,
            "spread": result.consensus_spread,
            "agreement": result.agreement,
            "summary": result.summary,
        },
        "models": [
            {
                "id": m.model_id,
                "name": m.model_name,
                "color": m.color,
                "probability": m.probability,
                "readings": m.readings,
                "spread": m.spread,
                "certainty": m.certainty,
                "signal": m.signal,
                "grounding": m.grounding,
                "grounding_note": m.grounding_note,
                "latency_ms": m.latency_ms,
                "error": m.error,
            }
            for m in result.models
        ],
        "meta": {
            "n_responded": result.n_responded,
            "n_total": result.n_total,
            "total_latency_ms": result.total_latency_ms,
        },
    }
