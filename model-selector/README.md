# model_selector

Find the **cheapest model that actually works** for a task. Prices and model lineups change
constantly; this keeps a local, versioned catalog and backs rankings with measured evidence.

```
discover ──► catalog.sqlite3 (snapshots, models, price history, probes)
                 │
probe suites ────┤  (easy/medium/hard prompt ladders per category, machine-graded)
                 │
rank "novel task" ──► cheapest → most expensive, pass-rate at task difficulty,
                      optional --trial cascade on your own sample
```

## Quick start

```bash
uv sync --extra dev          # or: uv pip install -e ".[dev]"
export OPENAI_API_KEY=...
uv run model-selector discover                  # docs pricing.md + models.md + /v1/models
uv run model-selector prices --category reasoning
uv run model-selector probe reasoning --efforts default low high --max-call-usd 0.05
uv run model-selector probe reasoning --unprobed-only        # just the new/unseen models
uv run model-selector stats --suite reasoning
uv run model-selector rank "extract fields from invoices to JSON" --sample @inv.txt --calls 50000
uv run model-selector rank "summarize support tickets" --sample @t.txt --trial --rubric "PASS if ..."
```

Data lives in `$MODEL_SELECTOR_HOME` (default `./data/model_selector`).

## How it works

- **Discovery** (`discovery/openai_docs.py`): fetches `…/api/docs/pricing.md` (docs serve markdown when
  `.md` is appended; HTML→markdown fallback), parses every price table incl. two-row short/long-context
  headers, tier tabs (Standard/Batch/Flex/Fast mode), multi-modality rows, and per-minute models.
  The model catalog page and `/v1/models` add models that have no listed price (**unseen**). Every run
  stores a content-hashed snapshot and reports `new_models`, `not_listed`, `price_changes`.
- **Taxonomy** (`taxonomy.py`): pricing-section → category, then model-id patterns. Override with
  `model-selector set-category <id> <cat>` (sticky across discoveries).
- **Unseen-model pricing**: sibling inference (`gpt-x-2026-05-18` → `gpt-x`), flagged `inferred:` in output.
- **Probes** (`probes/`): suites in `prompts/*.yaml`. Graders: exact / numeric / regex / json_fields /
  WER / code_exec (opt-in, subprocess) / llm_judge / vision_judge. Cost = real `usage` × current price
  (+ web-search call fees). Reasoning effort is a first-class dimension.
  - transcription: reference text → audio via cheapest TTS (cached) → transcribe → WER
  - tts: synthesize → round-trip through cheapest STT → WER
  - image: generate → vision judge checklist
  - deep_research: web_search tool, `--allow-expensive` required
  - realtime: **estimated only** from scenario assumptions in `prompts/realtime.yaml` (WebSocket probe TODO)
- **Ranking** (`ranker.py`): output tokens per call come from *observed* probe averages per
  model/effort, so reasoning-token inflation is priced in. Cheap-per-token @ high effort can lose to a
  pricier model @ default — the tests encode exactly that case.

## Tests
`uv run pytest` — fully offline (fixture docs + fake OpenAI client).
