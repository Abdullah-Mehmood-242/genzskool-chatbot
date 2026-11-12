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


@pytest.mark.parametrize(
    ("kwargs", "missing"),
    [
        ({"reranker_provider": "cohere", "reranker_model": "m"}, "RERANKER_API_KEY"),
        ({"reranker_provider": "voyage", "reranker_api_key": "real-key-1", "reranker_model": ""}, "RERANKER_MODEL"),
        ({"reranker_provider": "cross_encoder"}, "RERANKER_MODEL"),
    ],
)
def test_reranker_requires_key_and_model(kwargs: dict[str, str], missing: str) -> None:
    with pytest.raises(ConfigurationError, match=missing):
        _load(embedding_provider="fake", **kwargs)


def test_reranker_defaults_off() -> None:
    s = _load(embedding_provider="fake")
    assert s.reranker_provider == "none"
    s2 = _load(embedding_provider="fake", reranker_provider="cross_encoder", reranker_model="ms-marco")
    assert s2.reranker_model == "ms-marco"
