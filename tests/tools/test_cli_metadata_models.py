from __future__ import annotations

import importlib
from pathlib import Path

import pytest

augment_registry = importlib.import_module("tools._shared.augment_registry")


def test_augment_metadata_model_normalises_sequences() -> None:
    model = augment_registry.AugmentMetadataModel.model_validate(
        {
            "path": Path("augment.yaml"),
            "payload": {
                "operations": {
                    "cli.run": {
                        "tags": ["cli", "admin", "cli"],
                        "x-handler": "tests.cli:run",
                        "x-env": ["KGF_ENV"],
                        "examples": ["kgf run"],
                        "x-codeSamples": [
                            {"lang": "bash", "source": "kgf run"},
                        ],
                    }
                },
                "x-tagGroups": [
                    {"name": "Commands", "tags": ["cli", "admin", "cli"]},
                ],
            },
        }
    )

    override = model.operation_override("cli.run")
    assert override is not None
    assert override.tags == ("cli", "admin")
    assert override.env == ("KGF_ENV",)
    assert override.examples == ("kgf run",)
    assert model.payload["operations"]["cli.run"]["x-env"] == ["KGF_ENV"]
    assert model.tag_groups[0].tags == ("cli", "admin")


def test_registry_metadata_model_validates_interfaces() -> None:
    model = augment_registry.RegistryMetadataModel.model_validate(
        {
            "path": Path("registry.yaml"),
            "interfaces": {
                "tools-cli": {
                    "entrypoint": "tests.cli:app",
                    "owner": "docs",
                    "operations": {
                        "run": {
                            "operation_id": "cli.run",
                            "handler": "tests.cli:run",
                            "tags": ["cli"],
                        }
                    },
                }
            },
        }
    )

    interface = model.interface("tools-cli")
    assert interface is not None
    assert interface.entrypoint == "tests.cli:app"
    assert interface.operations["run"].handler == "tests.cli:run"


def test_load_augment_reports_validation_errors(tmp_path: Path) -> None:
    augment_path = tmp_path / "augment.yaml"
    augment_path.write_text(
        """
        operations:
          cli.run:
            tags: invalid  # not a sequence
        """,
        encoding="utf-8",
    )

    with pytest.raises(augment_registry.AugmentRegistryValidationError) as excinfo:
        augment_registry.load_augment(augment_path)

    problem = excinfo.value.problem
    assert problem["status"] == 422
    assert any("tags" in error.get("loc", "") for error in problem.get("errors", []))
