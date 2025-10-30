"""Tests for SQLite catalogue helpers."""

from __future__ import annotations

import json
from pathlib import Path

from kgfoundry.agent_catalog.models import AgentCatalogModel, load_catalog_model
from kgfoundry.agent_catalog.sqlite import load_catalog_from_sqlite, write_sqlite_catalog

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json")


def _load_fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_sqlite_round_trip(tmp_path: Path) -> None:
    """The SQLite writer should persist a payload that can be reloaded."""
    payload = _load_fixture_payload()
    sqlite_path = tmp_path / "catalog.sqlite"
    write_sqlite_catalog(payload, sqlite_path)
    loaded = load_catalog_from_sqlite(sqlite_path)
    model = AgentCatalogModel.model_validate(loaded)
    symbol = model.packages[0].modules[0].symbols[0]
    assert symbol.qname == "demo.module.fn"
    assert symbol.anchors.start_line == 10


def test_load_catalog_prefers_sqlite(tmp_path: Path) -> None:
    """``load_catalog_model`` should prefer SQLite artefacts when available."""
    payload = _load_fixture_payload()
    sqlite_path = tmp_path / "catalog.sqlite"
    write_sqlite_catalog(payload, sqlite_path)
    json_path = tmp_path / "agent_catalog.json"
    json_path.write_text("{}", encoding="utf-8")
    model = load_catalog_model(json_path)
    assert model.packages[0].modules[0].symbols[0].qname == "demo.module.fn"
