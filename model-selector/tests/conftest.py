from pathlib import Path

import pytest

from model_selector.config import Settings
from model_selector.discovery import OpenAIDiscovery
from model_selector.store import Store

FIX = Path(__file__).parent / "fixtures"


def fake_fetch(url: str) -> str:
    if "pricing" in url:
        return (FIX / "pricing_sample.md").read_text()
    if "models" in url:
        return (FIX / "models_sample.md").read_text()
    raise RuntimeError(url)


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path).ensure()


@pytest.fixture
def store(settings):
    st = Store(settings.db_path)
    st.ingest(OpenAIDiscovery(settings, fetch=fake_fetch,
                              list_api_models=lambda: ["gpt-6-luna", "gpt-6-luna-2026-05-18", "gpt-7-nova"]).run())
    return st
