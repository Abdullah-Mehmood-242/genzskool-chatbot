"""Embedding clients: one contract suite every provider must pass (LSP), plus provider specifics."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
import respx

from common.embeddings import (
    Embedder,
    FakeEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    build_embedder,
    providers,
    tokenize,
)
from common.errors import ConfigurationError, ProviderError, TransientProviderError, UpstreamTimeout

DIM = 16
LIBS_DIR = Path(__file__).resolve().parents[2]
KEY = "-".join(["test", "embedding", "key", "123"])  # built at runtime: no key-like literal in git


def _vectors_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    n = len(body["input"])
    # Return out of order to prove the client sorts by index.
    data = [{"index": i, "embedding": [float(i + 1)] * DIM} for i in reversed(range(n))]
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": n}})


def _http(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> Callable[[], Embedder]:
    def make() -> Embedder:
        router = respx.Router(base_url=cls.default_base_url)
        router.post("/embeddings").mock(side_effect=_vectors_response)
        return cls(api_key=KEY, model="m", dim=DIM, transport=httpx.MockTransport(router.handler))

    return make


FACTORIES: dict[str, Callable[[], Embedder]] = {
    "fake": lambda: FakeEmbedder(DIM),
    "voyage": _http(VoyageEmbedder),
    "openai": _http(OpenAIEmbedder),
}


@pytest.fixture(params=sorted(FACTORIES))
async def embedder(request: pytest.FixtureRequest) -> AsyncIterator[Embedder]:
    e = FACTORIES[request.param]()
    yield e
    await e.aclose()


# ---------- contract: every provider ----------


async def test_contract_one_vector_per_text_of_dim(embedder: Embedder) -> None:
    vectors = await embedder.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == DIM for v in vectors)
    assert all(isinstance(x, float) for v in vectors for x in v)


async def test_contract_empty_input_makes_no_call(embedder: Embedder) -> None:
    assert await embedder.embed_documents([]) == []


async def test_contract_query_vector(embedder: Embedder) -> None:
    v = await embedder.embed_query("shopify")
    assert len(v) == DIM


async def test_contract_name_and_dim(embedder: Embedder) -> None:
    assert embedder.name in providers()
    assert embedder.dim == DIM


# ---------- fake ----------


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_fake_is_normalised_and_deterministic() -> None:
    e = FakeEmbedder(256)
    a1, a2 = await e.embed_documents(["Shopify store setup", "Shopify store setup"])
    assert a1 == a2
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)


def test_fake_is_stable_across_processes() -> None:
    """Ingestion and retrieval run in different processes; hashing must not be salted."""
    import asyncio

    here = asyncio.run(FakeEmbedder(64).embed_query("TikTok Shop live selling"))
    code = (
        "import asyncio, json, sys; sys.path.insert(0, sys.argv[1]);"
        "from common.embeddings import FakeEmbedder;"
        "print(json.dumps(asyncio.run(FakeEmbedder(64).embed_query('TikTok Shop live selling'))))"
    )
    libs = str(LIBS_DIR)
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, libs],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "123"},
    )
    assert json.loads(out.stdout) == here


async def test_fake_similarity_is_lexical() -> None:
    e = FakeEmbedder(1024)
    shopify, shopify_q, graphic = await e.embed_documents(
        [
            "Shopify store setup to first sale, dropshipping",
            "how do I set up a shopify store",
            "Photoshop and Illustrator",
        ]
    )
    assert _cos(shopify, shopify_q) > _cos(shopify, graphic)
    assert _cos(shopify, shopify_q) > 0.2


@pytest.mark.parametrize("text", ["", "   ", "!!!", "؟؟"])
async def test_fake_handles_textless_input(text: str) -> None:
    v = await FakeEmbedder(32).embed_query(text)
    assert math.isclose(sum(x * x for x in v), 1.0)


async def test_fake_query_equals_document_embedding() -> None:
    e = FakeEmbedder(32)
    assert await e.embed_query("ebay") == (await e.embed_documents(["ebay"]))[0]


def test_fake_rejects_tiny_dim() -> None:
    with pytest.raises(ConfigurationError):
        FakeEmbedder(4)


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("AI automation with ChatGPT and Make.com", ["ai", "automation", "with", "chatgpt", "and", "make.com"]),
        ("Rs. 30,000", ["rs", "30", "000"]),
        ("Don't", ["don't"]),
    ],
)
def test_tokenize(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


# ---------- HTTP providers ----------


def _client(
    cls: type[VoyageEmbedder] | type[OpenAIEmbedder], handler: Callable[[httpx.Request], httpx.Response]
) -> Embedder:
    router = respx.Router(base_url=cls.default_base_url)
    route = router.post("/embeddings").mock(side_effect=handler)
    e = cls(api_key=KEY, model="model-x", dim=DIM, transport=httpx.MockTransport(router.handler))
    del route
    return e


HTTP = [VoyageEmbedder, OpenAIEmbedder]


@pytest.mark.parametrize("cls", HTTP)
async def test_http_sends_auth_and_model_and_sorts_by_index(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _vectors_response(request)

    e = _client(cls, handler)
    vectors = await e.embed_documents(["x", "y"])
    assert vectors[0][0] == 1.0
    assert vectors[1][0] == 2.0
    req = seen[0]
    assert req.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "model-x"
    assert body["input"] == ["x", "y"]


async def test_voyage_distinguishes_query_and_document() -> None:
    types: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        types.append(body["input_type"])
        assert body["output_dimension"] == DIM
        return _vectors_response(request)

    e = _client(VoyageEmbedder, handler)
    await e.embed_documents(["a"])
    await e.embed_query("b")
    assert types == ["document", "query"]


async def test_openai_requests_configured_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["dimensions"] == DIM
        return _vectors_response(request)

    await _client(OpenAIEmbedder, handler).embed_documents(["a"])


@pytest.mark.parametrize("cls", HTTP)
@pytest.mark.parametrize(
    ("status", "exc"),
    [
        (429, TransientProviderError),
        (500, TransientProviderError),
        (503, TransientProviderError),
        (401, ProviderError),
        (403, ProviderError),
        (400, ProviderError),
        (422, ProviderError),
    ],
)
async def test_http_status_mapping(
    cls: type[VoyageEmbedder] | type[OpenAIEmbedder], status: int, exc: type[ProviderError]
) -> None:
    e = _client(cls, lambda r: httpx.Response(status, json={"error": f"echo {KEY}"}))
    with pytest.raises(exc) as info:
        await e.embed_documents(["a"])
    assert type(info.value) is exc  # 4xx must NOT be transient: retrying a bad key is pointless
    assert KEY not in str(info.value)
    assert KEY not in str(info.value.details)


@pytest.mark.parametrize("cls", HTTP)
async def test_http_timeout_and_network_errors(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    def refused(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(UpstreamTimeout):
        await _client(cls, timeout).embed_documents(["a"])
    with pytest.raises(TransientProviderError):
        await _client(cls, refused).embed_documents(["a"])


@pytest.mark.parametrize("cls", HTTP)
@pytest.mark.parametrize(
    "payload",
    [{"nope": 1}, {"data": [{"index": 0}]}, {"data": [{"embedding": [1.0] * DIM}]}, {"data": "x"}],
)
async def test_http_malformed_response(cls: type[VoyageEmbedder] | type[OpenAIEmbedder], payload: object) -> None:
    with pytest.raises(ProviderError, match="malformed"):
        await _client(cls, lambda r: httpx.Response(200, json=payload)).embed_documents(["a"])


@pytest.mark.parametrize("cls", HTTP)
async def test_http_non_json_response(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    with pytest.raises(ProviderError, match="malformed"):
        await _client(cls, lambda r: httpx.Response(200, text="<html>")).embed_documents(["a"])


@pytest.mark.parametrize("cls", HTTP)
async def test_http_wrong_dimension_is_explained(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    resp = {"data": [{"index": 0, "embedding": [0.1] * (DIM + 1)}]}
    with pytest.raises(ProviderError, match="EMBEDDING_DIM"):
        await _client(cls, lambda r: httpx.Response(200, json=resp)).embed_documents(["a"])


@pytest.mark.parametrize("cls", HTTP)
async def test_http_wrong_count(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    resp = {"data": [{"index": 0, "embedding": [0.1] * DIM}]}
    with pytest.raises(ProviderError, match="1 vectors for 2"):
        await _client(cls, lambda r: httpx.Response(200, json=resp)).embed_documents(["a", "b"])


@pytest.mark.parametrize("cls", HTTP)
def test_http_requires_key_and_model(cls: type[VoyageEmbedder] | type[OpenAIEmbedder]) -> None:
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        cls(api_key="", model="m", dim=DIM)
    with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
        cls(api_key=KEY, model="", dim=DIM)


async def test_custom_base_url_for_openai_compatible() -> None:
    router = respx.Router(base_url="http://local-llm:8080/v1")
    router.post("/embeddings").mock(side_effect=_vectors_response)
    e = OpenAIEmbedder(
        api_key=KEY,
        model="m",
        dim=DIM,
        base_url="http://local-llm:8080/v1/",
        transport=httpx.MockTransport(router.handler),
    )
    assert len(await e.embed_documents(["a"])) == 1


# ---------- factory ----------


def test_factory_builds_each_provider() -> None:
    assert isinstance(build_embedder("fake", dim=DIM), FakeEmbedder)
    assert isinstance(build_embedder("voyage", api_key=KEY, model="m", dim=DIM), VoyageEmbedder)
    assert isinstance(build_embedder("openai", api_key=KEY, model="m", dim=DIM), OpenAIEmbedder)


def test_factory_unknown_provider() -> None:
    with pytest.raises(ConfigurationError, match="Unknown EMBEDDING_PROVIDER"):
        build_embedder("cohere", dim=DIM)


def test_sentence_transformers_missing_package_is_a_clear_config_error() -> None:
    pytest.importorskip("pytest")  # always present; documents intent
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ConfigurationError, match="sentence-transformers"):
            build_embedder("sentence_transformers", model="all-MiniLM-L6-v2", dim=384)
    else:  # pragma: no cover - only when the optional package is installed
        pytest.skip("sentence-transformers installed")


def test_providers_match_settings_literal() -> None:
    from typing import get_args

    from common.config import EmbeddingProviderName

    assert providers() == sorted(get_args(EmbeddingProviderName))
