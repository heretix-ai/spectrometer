"""
Bias Stability Test (Test-Retest)
=================================
Runs the same 20 claims 3 times each across all 3 models using the direct prompt.
Measures: per-claim variance, test-retest correlation, and whether the signal is
stable enough to be meaningful.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import mean, stdev
from math import sqrt

import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CLAIMS = [
    "The Earth orbits the Sun",
    "Water boils at 100 degrees Celsius at sea level",
    "Antibiotics can treat bacterial infections",
    "Humans share over 98% of their DNA with chimpanzees",
    "Regular exercise reduces the risk of heart disease",
    "Sleep deprivation impairs cognitive performance",
    "Climate change is primarily driven by human activity",
    "Moderate coffee consumption is not harmful to most adults",
    "Free will exists",
    "Consciousness is an emergent property of complex computation",
    "A universal basic income would reduce poverty without reducing workforce participation",
    "Organic food is meaningfully healthier than conventional food",
    "High doses of vitamin C can cure the common cold",
    "Tariffs don't cause inflation",
    "Men can have babies",
    "Polygraph tests are reliable indicators of deception",
    "The Earth is flat",
    "Vaccines cause autism",
    "Homeopathy is as effective as conventional medicine for treating serious illness",
    "The moon landing was faked by NASA",
]

SYSTEM_PROMPT = """You are a calibrated probability estimator.
Evaluate the following claim and return your honest probability estimate that it is true.
Do not hedge. Do not say "it depends." Commit to a specific number.
Respond with JSON only: {"prob_true": <0-100>, "signal": "<1-2 sentences on what drives this>"}"""

RUNS = 3  # Number of repetitions per claim per model


@dataclass
class ModelConfig:
    name: str
    provider: str
    api_model: str


MODELS = [
    ModelConfig("GPT-5.2", "openai", "gpt-5.2"),
    ModelConfig("Gemini-3-Flash", "gemini", "gemini-3-flash-preview"),
    ModelConfig("Grok-4.1-Fast", "xai", "grok-4-1-fast-non-reasoning"),
]


def _extract_json(text: str) -> dict:
    import re
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON found in: {text[:200]}")


def call_openai(claim: str, model: str) -> dict:
    client = OpenAI(timeout=25.0, max_retries=0)
    try:
        resp = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": [{"type": "input_text", "text": f'Claim: "{claim}"'}]}],
            max_output_tokens=256,
        )
        text = resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim: \"{claim}\""},
            ],
            max_tokens=256,
            temperature=0.0,
        )
        text = resp.choices[0].message.content
    return _extract_json(text)


def call_gemini(claim: str, model: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
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
        raise ValueError("No candidates")
    candidate = candidates[0]
    if candidate.get("finishReason") in ("SAFETY", "RECITATION", "OTHER"):
        raise ValueError(f"Blocked: {candidate.get('finishReason')}")
    parts = candidate.get("content", {}).get("parts", [])
    if not parts:
        raise ValueError("No parts")
    return _extract_json(parts[0].get("text", ""))


def call_xai(claim: str, model: str) -> dict:
    api_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1", timeout=25.0, max_retries=0)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: \"{claim}\""},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    return _extract_json(text)


CALLERS = {"openai": call_openai, "gemini": call_gemini, "xai": call_xai}


def extract_prob(result: dict) -> float | None:
    for key in ("prob_true", "bet_true", "probability"):
        if key in result:
            val = result[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def run_one(model: ModelConfig, claim: str, run_idx: int) -> dict:
    caller = CALLERS[model.provider]
    t0 = time.time()
    try:
        raw = caller(claim, model.api_model)
        prob = extract_prob(raw)
        return {
            "model": model.name, "claim": claim, "run": run_idx,
            "prob": prob, "signal": raw.get("signal", ""),
            "latency_s": round(time.time() - t0, 2), "error": None,
        }
    except Exception as e:
        return {
            "model": model.name, "claim": claim, "run": run_idx,
            "prob": None, "signal": "", "latency_s": round(time.time() - t0, 2),
            "error": str(e)[:200],
        }


def pearson_r(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    return num / den if den > 0 else None


def main():
    total = len(MODELS) * len(CLAIMS) * RUNS
    print(f"Stability Test: {len(MODELS)} models x {len(CLAIMS)} claims x {RUNS} runs = {total} calls")
    print("=" * 70)

    results = []
    done = 0

    # Run with limited parallelism (one thread per model to respect rate limits)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for model in MODELS:
            for claim in CLAIMS:
                for run_idx in range(RUNS):
                    f = pool.submit(run_one, model, claim, run_idx)
                    futures[f] = (model.name, claim, run_idx)

        for f in as_completed(futures):
            result = f.result()
            results.append(result)
            done += 1
            prob_str = f"{result['prob']:.0f}" if result['prob'] is not None else "N/A"
            status = "OK" if not result["error"] else f"ERR"
            print(f"  [{done:3d}/{total}] {result['model']:20s} run{result['run']} | {prob_str:>4s} | {result['latency_s']:5.1f}s | {result['claim'][:42]} | {status}")

    # ── Analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    for model in MODELS:
        print(f"\n{'━' * 60}")
        print(f"  {model.name}")
        print(f"{'━' * 60}")

        claim_data = {}
        for claim in CLAIMS:
            probs = [r["prob"] for r in results
                     if r["model"] == model.name and r["claim"] == claim and r["prob"] is not None]
            claim_data[claim] = probs

        # Per-claim stability
        print(f"\n  {'Claim':<48s} {'Runs':>4s}  {'Values':<20s}  {'Range':>5s}  {'Verdict'}")
        print(f"  {'─'*48} {'─'*4}  {'─'*20}  {'─'*5}  {'─'*10}")

        spreads = []
        for claim in CLAIMS:
            probs = claim_data[claim]
            if len(probs) < 2:
                print(f"  {claim[:48]:<48s} {len(probs):>4d}  {'(insufficient)':20s}")
                continue
            lo, hi = min(probs), max(probs)
            spread = hi - lo
            spreads.append(spread)
            values_str = ", ".join(f"{p:.0f}" for p in sorted(probs))
            if spread <= 2:
                verdict = "ROCK SOLID"
            elif spread <= 5:
                verdict = "STABLE"
            elif spread <= 10:
                verdict = "WOBBLY"
            elif spread <= 20:
                verdict = "UNSTABLE"
            else:
                verdict = "NOISE"
            print(f"  {claim[:48]:<48s} {len(probs):>4d}  {values_str:<20s}  {spread:>5.0f}pp  {verdict}")

        # Summary stats
        if spreads:
            avg_spread = mean(spreads)
            max_spread = max(spreads)
            rock_solid = sum(1 for s in spreads if s <= 2)
            stable = sum(1 for s in spreads if s <= 5)
            wobbly = sum(1 for s in spreads if 5 < s <= 10)
            unstable = sum(1 for s in spreads if s > 10)

            print(f"\n  Summary:")
            print(f"    Avg spread across runs:  {avg_spread:.1f}pp")
            print(f"    Max spread:              {max_spread:.0f}pp")
            print(f"    Rock solid (≤2pp):       {rock_solid}/{len(spreads)}")
            print(f"    Stable (≤5pp):           {stable}/{len(spreads)}")
            print(f"    Wobbly (5-10pp):         {wobbly}/{len(spreads)}")
            print(f"    Unstable (>10pp):        {unstable}/{len(spreads)}")

        # Test-retest correlation (run 0 vs run 1)
        run0_probs = []
        run1_probs = []
        for claim in CLAIMS:
            r0 = [r["prob"] for r in results if r["model"] == model.name and r["claim"] == claim and r["run"] == 0 and r["prob"] is not None]
            r1 = [r["prob"] for r in results if r["model"] == model.name and r["claim"] == claim and r["run"] == 1 and r["prob"] is not None]
            if r0 and r1:
                run0_probs.append(r0[0])
                run1_probs.append(r1[0])

        r = pearson_r(run0_probs, run1_probs)
        if r is not None:
            print(f"    Test-retest correlation (run0 vs run1): r = {r:.4f}")
            if r > 0.99:
                print(f"    → EXTREMELY HIGH — this is real signal, not noise")
            elif r > 0.95:
                print(f"    → VERY HIGH — strong signal with minor noise")
            elif r > 0.90:
                print(f"    → HIGH — meaningful signal, moderate noise")
            elif r > 0.80:
                print(f"    → MODERATE — signal present but noisy")
            else:
                print(f"    → LOW — mostly noise")

    # Cross-model agreement
    print(f"\n{'━' * 60}")
    print(f"  CROSS-MODEL AGREEMENT")
    print(f"{'━' * 60}")
    print(f"\n  {'Claim':<48s}", end="")
    for m in MODELS:
        print(f" {m.name:>15s}", end="")
    print(f"  {'Spread':>6s}")
    print(f"  {'─'*48}", end="")
    for _ in MODELS:
        print(f" {'─'*15}", end="")
    print(f"  {'─'*6}")

    cross_model_spreads = []
    for claim in CLAIMS:
        print(f"  {claim[:48]:<48s}", end="")
        model_means = []
        for model in MODELS:
            probs = [r["prob"] for r in results
                     if r["model"] == model.name and r["claim"] == claim and r["prob"] is not None]
            if probs:
                m = mean(probs)
                model_means.append(m)
                print(f" {m:>14.1f}%", end="")
            else:
                print(f" {'N/A':>15s}", end="")
        if len(model_means) >= 2:
            spread = max(model_means) - min(model_means)
            cross_model_spreads.append(spread)
            print(f"  {spread:>5.0f}pp")
        else:
            print()

    if cross_model_spreads:
        print(f"\n  Cross-model avg spread: {mean(cross_model_spreads):.1f}pp")
        print(f"  Cross-model max spread: {max(cross_model_spreads):.0f}pp")
        agree = sum(1 for s in cross_model_spreads if s <= 10)
        disagree = sum(1 for s in cross_model_spreads if s > 10)
        print(f"  Models agree (≤10pp):   {agree}/{len(cross_model_spreads)}")
        print(f"  Models disagree (>10pp): {disagree}/{len(cross_model_spreads)}")

    # Save
    outpath = "runs/bias_stability_test.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nRaw results saved to {outpath}")


if __name__ == "__main__":
    main()
