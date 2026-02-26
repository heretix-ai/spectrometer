# Bias Spectrometer

Extract training bias from AI models. Ask a claim, get probability estimates from GPT-5, Gemini, and Grok — see where they agree and where they don't.

## What it does

The Spectrometer queries multiple AI models with the same claim and extracts:

- **Probability estimate** (0-100%) — how likely the model thinks the claim is true
- **Grounding level** — whether the model has specific training data, is extrapolating, or has no relevant knowledge
- **Cross-model consensus** — agreement, spread, and summary across all models

Each model is called twice for stability verification. Total wall clock: ~4 seconds.

## Quick start

```bash
git clone https://github.com/heretix-ai/spectrometer.git
cd spectrometer
uv sync

# Set at least one API key
export OPENAI_API_KEY=sk-...

# Run
uv run python serve.py
```

Open http://localhost:8000

## Docker

```bash
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

## Models

| Model | Provider | API Key |
|-------|----------|---------|
| GPT-5.2 | OpenAI | `OPENAI_API_KEY` |
| Gemini 3 Flash | Google | `GEMINI_API_KEY` |
| Grok 4.1 Fast | xAI | `XAI_API_KEY` |

Only models with configured API keys will be used. Set one, two, or all three.

## How it works

1. Your claim is sent to each configured model with a calibrated probability prompt
2. Each model returns a 0-100 probability estimate, reasoning signal, and epistemic grounding
3. Cross-model consensus is computed (agreement level, spread)
4. Results are displayed on a gauge with per-model detail cards

The prompt asks models to be direct probability estimators — no hedging, no "it depends." This extracts training bias rather than the safety-tuned default responses.

## Stability

Test-retest reliability across 20 claims × 3 runs:
- GPT-5.2: r = 0.9915 (avg spread 3.4pp)
- Gemini 3 Flash: r = 1.0000 (avg spread 0.0pp)
- Grok 4.1 Fast: r = 0.9997 (avg spread 0.3pp)

Run the stability test yourself: `uv run python scripts/bias_stability_test.py`

## Hosted version

Don't want to manage API keys? Use the hosted version at [heretix.ai](https://heretix.ai).

## License

MIT
