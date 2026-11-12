from __future__ import annotations

from pathlib import Path

import pytest

from app.settings import Settings
from common.config import load_settings
from common.errors import ConfigurationError


def _load(**kw: object) -> Settings:
    return load_settings(Settings, _env_file=None, **kw)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("PDF_DIR", "VECTOR_STORE", "AUTO_INGEST_ON_STARTUP", "AUTO_INGEST_INTERVAL_S"):
        monkeypatch.delenv(var, raising=False)
    s = _load(embedding_provider="fake")
    assert s.service_name == "ingestion"
    assert s.pdf_dir == Path("data/pdfs")
    assert s.vector_store == "qdrant"
    assert s.auto_ingest_on_startup is True
    assert s.auto_ingest_interval_s == 30.0
    assert s.auto_ingest_prune is True


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
