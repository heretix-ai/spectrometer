"""
Bias Elicitation A/B Test
=========================
Compares two prompt strategies across 3 models x 20 claims:
  A) Direct ask: "probability 0-100 this is true?"
  B) Betting frame: "you have $100 in a truth market, allocate"

Measures: mean, stddev, range, and whether betting frame widens the
dynamic range (pulls probabilities away from 50%).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean, stdev

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Claims spanning the full truth spectrum ──────────────────────────
# Grouped roughly: clearly true, probably true, contested, probably false, clearly false
CLAIMS = [
    # Clearly true (~90-100%)
    "The Earth orbits the Sun",
    "Water boils at 100 degrees Celsius at sea level",
    "Antibiotics can treat bacterial infections",
    "Humans share over 98% of their DNA with chimpanzees",
    # Probably true (~65-85%)
    "Regular exercise reduces the risk of heart disease",
    "Sleep deprivation impairs cognitive performance",
    "Climate change is primarily driven by human activity",
    "Moderate coffee consumption is not harmful to most adults",
    # Genuinely contested (~35-65%)
    "Free will exists",
    "Consciousness is an emergent property of complex computation",
    "A universal basic income would reduce poverty without reducing workforce participation",
    "Organic food is meaningfully healthier than conventional food",
    # Probably false (~15-35%)
    "High doses of vitamin C can cure the common cold",
    "Tariffs don't cause inflation",
    "Men can have babies",
    "Polygraph tests are reliable indicators of deception",
    # Clearly false (~0-10%)
    "The Earth is flat",
    "Vaccines cause autism",
    "Homeopathy is as effective as conventional medicine for treating serious illness",
    "The moon landing was faked by NASA",
]

# ── Prompt strategies ────────────────────────────────────────────────

SYSTEM_DIRECT = """You are a calibrated probability estimator.
Evaluate the following claim and return your honest probability estimate that it is true.
Do not hedge. Do not say "it depends." Commit to a specific number.
Respond with JSON only: {"prob_true": <0-100>, "signal": "<1-2 sentences on what drives this>"}"""

SYSTEM_BETTING = """You are a calibrated forecaster in a high-stakes prediction market.
You have $100. You must allocate your entire balance between TRUE and FALSE for the claim below.
A 50/50 split is a losing strategy — the oracle always resolves definitively.

Calibration anchors:
  $0 on TRUE = "The moon is made of cheese" (absurd)
  $50 on TRUE = "It will rain in London on a random future date" (genuine coin flip)
  $100 on TRUE = "1+1=2" (certain)

Respond with JSON only: {"bet_true": <0-100>, "signal": "<1-2 sentences on what drives this>"}"""

# ── Model configs ────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    provider: str  # openai | gemini | xai
    api_model: str

MODELS = [
    ModelConfig("GPT-5.2", "openai", "gpt-5.2"),
    ModelConfig("Gemini-3-Flash", "gemini", "gemini-3-flash-preview"),
    ModelConfig("Grok-4.1-Fast", "xai", "grok-4-1-fast-non-reasoning"),
]

# ── Provider call functions ──────────────────────────────────────────

def call_openai(system: str, claim: str, model: str) -> dict:
    client = OpenAI(timeout=20.0, max_retries=0)
    try:
        # GPT-5.2 uses the Responses API
        resp = client.responses.create(
            model=model,
            instructions=system,
            input=[{"role": "user", "content": [{"type": "input_text", "text": f'Claim: "{claim}"'}]}],
            max_output_tokens=256,
        )
        text = resp.output_text
    except Exception:
        # Fallback to chat completions
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Claim: \"{claim}\""},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
    # Strip markdown code fences if present
    if text.strip().startswith("```"):
        lines = text.strip().split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


def _extract_json_from_text(text: str) -> dict:
    """Try to extract JSON from text that may have markdown fences or extra content."""
    import re
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {text[:200]}")


def call_gemini(system: str, claim: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"role": "system", "parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": f'Claim: "{claim}"'}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "max_output_tokens": 512,
        },
    }
    r = requests.post(url, params={"key": api_key}, json=payload, timeout=25)
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates in Gemini response: {json.dumps(data)[:300]}")
    candidate = candidates[0]
    # Check for safety blocks
    finish_reason = candidate.get("finishReason", "")
    if finish_reason in ("SAFETY", "RECITATION", "OTHER"):
        raise ValueError(f"Gemini blocked: finishReason={finish_reason}")
    content = candidate.get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError(f"No parts in Gemini response: {json.dumps(candidate)[:300]}")
    text = parts[0].get("text", "")
    return _extract_json_from_text(text)


def call_xai(system: str, claim: str, model: str) -> dict:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=20.0, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Claim: \"{claim}\""},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    # Grok sometimes wraps in markdown code blocks
    if text.strip().startswith("```"):
        lines = text.strip().split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


CALLERS = {
    "openai": call_openai,
    "gemini": call_gemini,
    "xai": call_xai,
}


def extract_prob(result: dict, strategy: str) -> float | None:
    """Pull the 0-100 number from whichever key the model used."""
    for key in ("bet_true", "prob_true", "probability", "bet_TRUE", "prob"):
        if key in result:
            val = result[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


# ── Run one (model, strategy, claim) ─────────────────────────────────

def run_one(model: ModelConfig, strategy: str, claim: str) -> dict:
    system = SYSTEM_BETTING if strategy == "betting" else SYSTEM_DIRECT
    caller = CALLERS[model.provider]
    t0 = time.time()
    try:
        raw = caller(system, claim, model.api_model)
        latency = time.time() - t0
        prob = extract_prob(raw, strategy)
        return {
            "model": model.name,
            "strategy": strategy,
            "claim": claim,
            "prob": prob,
            "signal": raw.get("signal", ""),
            "raw": raw,
            "latency_s": round(latency, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "model": model.name,
            "strategy": strategy,
            "claim": claim,
            "prob": None,
            "signal": "",
            "raw": None,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:200],
        }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print(f"Bias Elicitation A/B Test")
    print(f"Models: {', '.join(m.name for m in MODELS)}")
    print(f"Claims: {len(CLAIMS)}")
    print(f"Strategies: direct, betting")
    print(f"Total calls: {len(MODELS) * len(CLAIMS) * 2}")
    print("=" * 70)

    results = []
    total = len(MODELS) * len(CLAIMS) * 2
    done = 0

    # Run sequentially per model to respect rate limits, but parallelize across models
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for model in MODELS:
            for claim in CLAIMS:
                for strategy in ["direct", "betting"]:
                    f = pool.submit(run_one, model, strategy, claim)
                    futures[f] = (model.name, strategy, claim)

        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            done += 1
            status = "OK" if result["error"] is None else f"ERR: {result['error'][:50]}"
            prob_str = f"{result['prob']:.0f}" if result['prob'] is not None else "N/A"
            print(f"  [{done}/{total}] {result['model']:20s} | {result['strategy']:7s} | {prob_str:>4s} | {result['latency_s']:5.1f}s | {result['claim'][:40]}... | {status}")

    # ── Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for model in MODELS:
        print(f"\n{'─' * 50}")
        print(f"  {model.name}")
        print(f"{'─' * 50}")

        for strategy in ["direct", "betting"]:
            probs = [r["prob"] for r in results
                     if r["model"] == model.name and r["strategy"] == strategy and r["prob"] is not None]
            if len(probs) < 2:
                print(f"  {strategy:7s}: insufficient data ({len(probs)} valid)")
                continue

            avg = mean(probs)
            sd = stdev(probs)
            lo, hi = min(probs), max(probs)
            rng = hi - lo
            # Distance from 50 — measures how far from center the distribution sits
            avg_dist_from_50 = mean([abs(p - 50) for p in probs])

            print(f"  {strategy:7s}: mean={avg:5.1f}  sd={sd:5.1f}  range=[{lo:.0f}–{hi:.0f}] ({rng:.0f}pp)  avg_dist_from_50={avg_dist_from_50:.1f}")

        # Per-claim comparison
        print(f"\n  Per-claim comparison (betting - direct):")
        deltas = []
        for claim in CLAIMS:
            d = [r for r in results if r["model"] == model.name and r["claim"] == claim and r["strategy"] == "direct" and r["prob"] is not None]
            b = [r for r in results if r["model"] == model.name and r["claim"] == claim and r["strategy"] == "betting" and r["prob"] is not None]
            if d and b:
                dp, bp = d[0]["prob"], b[0]["prob"]
                delta = bp - dp
                # Did betting push AWAY from 50?
                d_dist = abs(dp - 50)
                b_dist = abs(bp - 50)
                wider = "WIDER" if b_dist > d_dist else "narrower" if b_dist < d_dist else "same"
                deltas.append({"claim": claim, "direct": dp, "betting": bp, "delta": delta, "wider": wider})
                print(f"    {claim[:45]:45s}  direct={dp:5.1f}  betting={bp:5.1f}  delta={delta:+5.1f}  {wider}")

        if deltas:
            wider_count = sum(1 for d in deltas if d["wider"] == "WIDER")
            narrower_count = sum(1 for d in deltas if d["wider"] == "narrower")
            print(f"\n  VERDICT: Betting pushed WIDER on {wider_count}/{len(deltas)} claims, narrower on {narrower_count}/{len(deltas)}")

    # Save raw results
    outpath = "runs/bias_elicitation_test.json"
    os.makedirs("runs", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {outpath}")


if __name__ == "__main__":
    main()
