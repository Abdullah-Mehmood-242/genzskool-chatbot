from __future__ import annotations

import pytest

from app.settings import Settings
from common.config import load_settings
from common.errors import ConfigurationError


def _load(**kw: object) -> Settings:
    return load_settings(Settings, _env_file=None, **kw)


def test_defaults() -> None:
    s = _load(embedding_provider="fake")
    assert s.service_name == "retrieval"
    assert s.qdrant_collection == "genzskool_kb"


def test_openai_embeddings_need_key() -> None:
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        _load(embedding_provider="openai", embedding_api_key="YOUR_EMBEDDING_API_KEY_HERE", embedding_model="m")
