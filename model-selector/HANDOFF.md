# Handoff: integrating model_selector into Quindral

Built outside the repo (no access to /Users/vmac/prog/PY/Quindral). Status: 10/10 offline tests pass;
live discovery NOT yet run against developers.openai.com (sandbox had no egress) — run it first.

## First steps in the repo
1. Read `PROJECT_OVERVIEW.md`; decide placement:
   - as a package: copy `src/model_selector/` → `Quindral/<pkg_root>/model_selector/`, merge deps
     (`httpx openai pyyaml markdownify`) into Quindral's pyproject via `uv add`, move tests.
   - or keep standalone and `uv add --editable ../model-selector`.
2. Point `MODEL_SELECTOR_HOME` at Quindral's data dir convention; add the sqlite file to .gitignore.
3. `model-selector discover` → inspect JSON report. If `n_prices` is 0 or categories look wrong,
   the live page format differs from `tests/fixtures/pricing_sample.md`: save the live `.md` as a new
   fixture and fix `discovery/mdtables.py` / `taxonomy.SECTION_MAP`.
4. Cheap first probe: `probe reasoning --models gpt-6-luna --max-difficulty easy`.

## Known gaps / next work
- Realtime probing (WebSocket) — currently scenario estimates; calibrate token/sec assumptions.
- Deep research: models page currently lists no dedicated deep-research model; category is empty until
  one appears (discovery will pick it up via id pattern `deep-research`) — or map a flagship+web_search.
- Tier choice in ranking (Batch/Flex are ~50% off for async work) — add `--tier` to `rank`.
- Long-context price switch: set `Settings.long_context_threshold` once the threshold is confirmed.
- Provider abstraction: `Discovery` protocol + adapters are provider-agnostic; add Anthropic/Google next.
- Integration hook for Quindral's pipelines: `from model_selector.ranker import rank, TaskSpec`.
