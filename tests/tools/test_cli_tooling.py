from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

import pytest

cli_tooling = importlib.import_module("tools._shared.cli_tooling")


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")


def test_load_cli_tooling_context_success(tmp_path: Path) -> None:
    augment_path = tmp_path / "augment.yaml"
    registry_path = tmp_path / "registry.yaml"

    _write_yaml(
        augment_path,
        """
        tags:
          - name: orchestration
            description: Example tag
        operations:
          cli.run:
            tags: [orchestration]
        """,
    )

    _write_yaml(
        registry_path,
        """
        interfaces:
          tools-cli:
            entrypoint: tests.fixtures.cli:app
            binary: kgf
            operations: {}
        """,
    )

    settings = cli_tooling.CLIToolSettings(
        bin_name="kgf",
        title="Test CLI",
        version="1.0.0",
        augment_path=augment_path,
        registry_path=registry_path,
        interface_id="tools-cli",
    )

    context = cli_tooling.load_cli_tooling_context(settings)

    cli_config = context.cli_config
    assert cli_config.bin_name == "kgf"
    assert cli_config.interface_meta is not None
    assert cli_config.interface_meta.entrypoint == "tests.fixtures.cli:app"
    override = context.augment.operation_override("cli.run")
    assert override is not None
    assert override.tags == ("orchestration",)


def test_load_cli_tooling_context_missing_augment(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    _write_yaml(registry_path, "interfaces: {}")

    settings = cli_tooling.CLIToolSettings(
        bin_name="kgf",
        title="Broken CLI",
        version="0.0.1",
        augment_path=tmp_path / "missing.yaml",
        registry_path=registry_path,
    )

    with pytest.raises(cli_tooling.CLIConfigError) as excinfo:
        cli_tooling.load_cli_tooling_context(settings)

    problem = excinfo.value.problem
    assert problem["status"] == 404
    assert problem["type"] == "https://kgfoundry.dev/problems/cli-config"


def test_load_cli_tooling_context_missing_interface(tmp_path: Path) -> None:
    augment_path = tmp_path / "augment.yaml"
    registry_path = tmp_path / "registry.yaml"

    _write_yaml(augment_path, "operations: {}")
    _write_yaml(registry_path, "interfaces: {}")

    settings = cli_tooling.CLIToolSettings(
        bin_name="kgf",
        title="Test",
        version="0.0.1",
        augment_path=augment_path,
        registry_path=registry_path,
        interface_id="missing-cli",
    )

    with pytest.raises(cli_tooling.CLIConfigError) as excinfo:
        cli_tooling.load_cli_tooling_context(settings)

    problem = excinfo.value.problem
    assert problem["status"] == 422
    assert problem["detail"].startswith("Interface 'missing-cli'")


def test_loaders_use_caching(tmp_path: Path) -> None:
    augment_path = tmp_path / "augment.yaml"
    registry_path = tmp_path / "registry.yaml"

    _write_yaml(augment_path, "operations: {}")
    _write_yaml(registry_path, "interfaces: {}")

    first_augment = cli_tooling.load_augment_config(augment_path)
    second_augment = cli_tooling.load_augment_config(augment_path)
    assert first_augment is second_augment

    first_registry = cli_tooling.load_registry_context(registry_path)
    second_registry = cli_tooling.load_registry_context(registry_path)
    assert first_registry is second_registry
