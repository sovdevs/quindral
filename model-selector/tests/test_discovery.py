from model_selector.discovery.mdtables import parse_price, parse_tables
from conftest import FIX


def test_price_cells():
    assert parse_price("$10.00") == (10.0, None)
    assert parse_price("$0.034 / minute") == (0.034, "minute")
    assert parse_price("-") == (None, None)


def test_tiers_and_two_row_header():
    t = parse_tables((FIX / "pricing_sample.md").read_text())
    flag = [x for x in t if x.section == "Flagship models"]
    assert [x.tier for x in flag] == ["Standard", "Batch"]
    assert "Short context Input" in flag[0].columns and "Long context Output" in flag[0].columns
    img = [x for x in t if x.section == "Image generation models"]
    assert [x.tier for x in img] == ["Standard", "Standard"]  # disjoint model list, not Batch
    spec = [x for x in t if x.section == "Specialized models"]
    assert [x.tier for x in spec] == ["Standard", "Fast mode"]


def test_ingest(store):
    cats = {m["model_id"]: m["category"] for m in store.models()}
    assert cats["gpt-6-luna"] == "reasoning"
    assert cats["gpt-5.3-codex"] == "coding"
    assert cats["gpt-transcribe"] == "transcription"
    assert cats["gpt-live-transcribe"] == "realtime_transcription"
    assert cats["gpt-realtime-translate"] == "realtime_translation"
    assert cats["gpt-4o-mini-tts"] == "tts"          # from catalog page only
    assert cats["gpt-image-2"] == "image_generation"
    assert cats["gpt-7-nova"] == "reasoning"         # unseen, from /v1/models
    assert "o4-mini-2025-04-16" not in cats           # finetuning skipped
    luna = {r["tier"]: r for r in store.current_prices("openai", "gpt-6-luna")}
    assert luna["Standard"]["input"] == 0.10 and luna["Standard"]["long_output"] == 0.75
    assert luna["Batch"]["input"] == 0.05
    rt = {r["modality"]: r for r in store.current_prices("openai", "gpt-realtime-2.1")}
    assert rt["audio"]["input"] == 32.0 and rt["text"]["output"] == 24.0
    tr = store.current_prices("openai", "gpt-transcribe")[0]
    assert tr["per_unit"] == 0.0045 and tr["unit"] == "minute"


def test_sibling_price_inference(store):
    row, basis = store.price_for("openai", "gpt-6-luna-2026-05-18")
    assert basis == "inferred:gpt-6-luna" and row["input"] == 0.10
    assert store.price_for("openai", "gpt-7-nova")[1] == "unknown"


def test_price_change_detection(store, settings):
    from conftest import fake_fetch
    from model_selector.discovery import OpenAIDiscovery

    def changed(url):
        return fake_fetch(url).replace("| gpt-6-luna  | $0.10 ", "| gpt-6-luna  | $0.08 ")
    rep = store.ingest(OpenAIDiscovery(settings, fetch=changed, list_api_models=lambda: []).run())
    ch = [c for c in rep["price_changes"] if c["model"] == "gpt-6-luna" and c["tier"] == "Standard"]
    assert ch and ch[0]["new"]["input"] == 0.08
    assert "gpt-7-nova" in rep["not_listed"]
