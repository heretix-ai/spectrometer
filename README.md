# Bias Spectrometer

Extract training bias from AI models. Ask a claim, get probability estimates from GPT-5, Gemini, and Grok — see where they agree and where they don't.

## What it does

The Spectrometer queries multiple AI models with the same claim and extracts:

- **Probability estimate** (0-100%) — how likely the model thinks the claim is true
- **Grounding level** — whether the model has specific training data, is extrapolating, or has no relevant knowledge
- **Cross-model consensus** — agreement, spread, and summary across all models

Each model is called twice for stability verification. Total wall clock: ~4 seconds.

## Quick start (local)

```bash
# Clone
git clone https://github.com/heretix-ai/spectrometer.git
cd spectrometer

# Install
uv sync

# Set at least one API key
export OPENAI_API_KEY=sk-...

# Start Postgres
docker compose up -d postgres

# Run migrations
DATABASE_URL=postgresql+psycopg://spectrometer:spectrometer@localhost:5433/spectrometer uv run alembic upgrade head

# Start the server
uv run uvicorn api.main:app --reload
```

Open http://localhost:8000 in your browser.

## Self-host (Docker)

```bash
cp .env.example .env
# Edit .env with your API keys
docker compose up -d
```

## Models

| Model | Provider | Color |
|-------|----------|-------|
| GPT-5.2 | OpenAI | Green |
| Gemini 3 Flash | Google | Indigo |
| Grok 4.1 Fast | xAI | Amber |

Configure API keys via environment variables: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`.

## Hosted version

https://heretix.ai — managed hosting with auth, usage tracking, and Stripe billing.

## License

MIT
