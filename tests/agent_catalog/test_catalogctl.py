from pathlib import Path

from tools.agent_catalog import catalogctl

CATALOG_PATH = Path(__file__).resolve().parents[2] / "docs" / "_build" / "agent_catalog.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_capabilities_command(capsys) -> None:
    exit_code = catalogctl.main(
        [
            "--catalog",
            str(CATALOG_PATH),
            "--repo-root",
            str(REPO_ROOT),
            "capabilities",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "kgfoundry" in captured.out


def test_search_command(capsys) -> None:
    exit_code = catalogctl.main(
        [
            "--catalog",
            str(CATALOG_PATH),
            "--repo-root",
            str(REPO_ROOT),
            "search",
            "catalog",
            "--k",
            "2",
        ]
    )
    assert exit_code == 0
    payload = capsys.readouterr().out
    assert "lexical_score" in payload
