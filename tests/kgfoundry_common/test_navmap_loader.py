from __future__ import annotations

from collections.abc import Iterable

from kgfoundry_common.navmap_loader import NavMetadataModel, load_nav_metadata


def _symbol_names(sections: Iterable[dict[str, object]] | None) -> set[str]:
    names: set[str] = set()
    if sections is None:
        return names
    for section in sections:
        symbols = section.get("symbols", []) if isinstance(section, dict) else []
        if isinstance(symbols, Iterable):
            for symbol in symbols:
                if isinstance(symbol, str):
                    names.add(symbol)
    return names


def test_cli_nav_metadata_derives_from_cli_contracts() -> None:
    metadata = load_nav_metadata("download.cli", ("app", "harvest"))
    assert isinstance(metadata, NavMetadataModel)
    assert metadata.module_meta.owner == "@data-platform"
    assert metadata.symbols["harvest"].handler is not None
    # Sections should reflect CLI tag groups and include harvest symbol.
    section_symbol_names = _symbol_names(metadata["sections"])
    assert "harvest" in section_symbol_names


def test_sidecar_metadata_validates() -> None:
    metadata = load_nav_metadata("registry.helper", ("DuckDBRegistryHelper",))
    assert isinstance(metadata, NavMetadataModel)
    assert metadata.title == "registry"
    assert "DuckDBRegistryHelper" in metadata.symbols
