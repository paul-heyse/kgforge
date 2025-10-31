from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from fastapi.testclient import TestClient
from httpx import Response

from kgfoundry.search_api.app import app
from kgfoundry.search_client import KGFoundryClient
from kgfoundry.search_client.client import SupportsHttp, SupportsResponse


class _RecordingHttp(SupportsHttp):
    """Test double that records requests while delegating to TestClient."""

    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(
        self,
        method: str,
        url: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[str, tuple[object, ...], dict[str, object]]:
        self.records.append((method, args, kwargs))
        return url, args, kwargs

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        url, args, kwargs = self._record("GET", url, args, kwargs)
        client_get = cast(Callable[..., Response], self._client.get)
        response = client_get(url, *args, **cast(dict[str, Any], kwargs))
        return cast(SupportsResponse, response)

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        url, args, kwargs = self._record("POST", url, args, kwargs)
        client_post = cast(Callable[..., Response], self._client.post)
        response = client_post(url, *args, **cast(dict[str, Any], kwargs))
        return cast(SupportsResponse, response)


class _StubResponse(SupportsResponse):
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload: dict[str, object] = dict(payload)
        self.raise_calls = 0

    def raise_for_status(self) -> None:
        self.raise_calls += 1

    def json(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.payload.copy())


class _StubHttp(SupportsHttp):
    def __init__(self, response: _StubResponse) -> None:
        self.response = response

    def get(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        del url, args, kwargs
        return self.response

    def post(self, url: str, /, *args: object, **kwargs: object) -> SupportsResponse:
        del url, args, kwargs
        return self.response


def test_client_calls_api() -> None:
    client = TestClient(app)
    http = _RecordingHttp(client)
    c = KGFoundryClient(base_url=str(client.base_url), http=http)

    assert c.healthz()["status"] == "ok"
    res = c.search("test", k=3)
    assert "results" in res
    assert any(record[0] == "GET" for record in http.records)
    assert any(record[0] == "POST" for record in http.records)


def test_client_calls_raise_for_status() -> None:
    response = _StubResponse({"status": "ok"})
    client = KGFoundryClient(base_url="http://example.com", http=_StubHttp(response))
    client.healthz()
    assert response.raise_calls == 1
