"""Tests for the analytics builder utilities."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from tools.docs import build_agent_analytics as analytics


def test_catalog_metrics_counts_packages_modules_and_symbols() -> None:
    catalog = {
        "packages": [
            {
                "modules": [
                    {"qualified": "pkg.mod", "symbols": ["a", "b"]},
                    {"qualified": "pkg.other", "symbols": []},
                ]
            },
            {"modules": [{"qualified": "pkg.sub", "symbols": ["x"]}]},
            "not-a-package",
        ],
        "shards": {"packages": [1, 2]},
    }

    metrics = analytics._catalog_metrics(cast(analytics.JSONMapping, catalog))

    assert metrics.packages == 2
    assert metrics.modules == 3
    assert metrics.symbols == 3
    assert metrics.shards == 2


def test_check_links_reports_missing_sources_and_pages(tmp_path: Path) -> None:
    repo_root = tmp_path
    existing_source = repo_root / "src" / "pkg" / "mod.py"
    existing_source.parent.mkdir(parents=True, exist_ok=True)
    existing_source.write_text("# module", encoding="utf-8")

    existing_page = repo_root / "docs" / "pkg" / "mod.md"
    existing_page.parent.mkdir(parents=True, exist_ok=True)
    existing_page.write_text("# docs", encoding="utf-8")

    catalog = {
        "packages": [
            {
                "modules": [
                    {
                        "qualified": "pkg.mod",
                        "source": {"path": "src/pkg/mod.py"},
                        "pages": {"overview": "docs/pkg/mod.md"},
                    },
                    {
                        "qualified": "pkg.other",
                        "source": {"path": "src/pkg/other.py"},
                        "pages": {"api": "docs/pkg/other.md"},
                    },
                ]
            }
        ]
    }

    issues = analytics._check_links(cast(analytics.JSONMapping, catalog), repo_root, sample=10)

    assert any(detail.path == "src/pkg/other.py" for detail in issues)
    assert any(detail.page == "docs/pkg/other.md" for detail in issues)


def test_legacy_to_document_round_trips_core_fields() -> None:
    payload = {
        "generated_at": "2024-01-01T00:00:00Z",
        "repo": {"root": "/workspace"},
        "catalog": {"packages": "3", "modules": 5, "symbols": "7", "shards": 1},
        "portal": {"sessions": {"builds": "2", "unique_users": 4}},
        "errors": {
            "broken_links": "5",
            "details": [
                {
                    "module": "pkg.mod",
                    "path": "missing.py",
                    "page": "docs/missing.md",
                    "kind": "api",
                },
                {"module": "pkg.other"},
            ],
        },
    }

    document = analytics._legacy_to_document(cast(analytics.JSONMapping, payload))

    assert document.generatedAt == "2024-01-01T00:00:00Z"
    assert document.repo.root == "/workspace"
    assert document.catalog.packages == 3
    assert document.catalog.modules == 5
    assert document.catalog.symbols == 7
    assert document.catalog.shards == 1
    assert document.portal.sessions.builds == 2
    assert document.portal.sessions.unique_users == 4
    assert document.errors.broken_links == 5
    assert len(document.errors.details) == 2
