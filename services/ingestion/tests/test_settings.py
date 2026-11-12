from __future__ import annotations

from pathlib import Path

import pytest

from app.settings import Settings
from common.config import load_settings
from common.errors import ConfigurationError


def _load(**kw: object) -> Settings:
    return load_settings(Settings, _env_file=None, **kw)


def test_defaults() -> None:
    s = _load(embedding_provider="fake")
    assert s.service_name == "ingestion"
    assert s.pdf_dir == Path("data/pdfs")
    assert s.vector_store == "qdrant"


def test_real_embedding_provider_needs_key_and_model() -> None:
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        _load(embedding_provider="voyage", embedding_model="voyage-x")
    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
        _load(
            embedding_provider="voyage",
            embedding_api_key="pa-real-key",
            embedding_model="YOUR_EMBEDDING_MODEL_NAME_HERE",
        )


def test_local_sentence_transformers_needs_model_but_no_key() -> None:
    s = _load(embedding_provider="sentence_transformers", embedding_model="all-MiniLM-L6-v2")
    assert s.embedding_provider == "sentence_transformers"


@pytest.mark.parametrize(
    ("field", "value"), [("qdrant_collection", "bad name!"), ("embedding_dim", 2), ("vector_store", "pinecone")]
)
def test_invalid_values_fail_fast(field: str, value: object) -> None:
    with pytest.raises(ConfigurationError, match=field):
        _load(**{field: value})
