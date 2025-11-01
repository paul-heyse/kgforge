"""Generate MkDocs API reference pages from the Griffe symbol graph."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import mkdocs_gen_files
from tools import get_logger, with_fields

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from docs._scripts import shared as shared_module
else:
    shared_module = importlib.import_module("docs._scripts.shared")

DocsSettings = shared_module.DocsSettings
GriffeLoader = shared_module.GriffeLoader
detect_environment = shared_module.detect_environment
ensure_sys_paths = shared_module.ensure_sys_paths
load_settings = shared_module.load_settings
make_loader = shared_module.make_loader


class GriffeNode(Protocol):
    """Protocol representing the subset of Griffe objects we rely on."""

    path: str
    members: Mapping[str, GriffeNode]
    is_package: bool
    is_module: bool


ENV = detect_environment()
ensure_sys_paths(ENV)
ROOT = ENV.root

LOGGER = get_logger(__name__)
LOG = with_fields(LOGGER, operation="mkdocs_api")

SETTINGS: DocsSettings = load_settings()
LOADER: GriffeLoader = make_loader(ENV)


def iter_packages() -> list[str]:
    """Return the packages that should receive API documentation pages."""
    return list(SETTINGS.packages)


def _write_index(destination: Path) -> None:
    """Write the API reference landing page."""
    with mkdocs_gen_files.open(destination / "index.md", "w") as handle:
        handle.write("# API Reference\n")


def _write_node(destination: Path, node: GriffeNode) -> None:
    """Render a single MkDocs page for ``node``."""
    rel = node.path.replace(".", "/")
    page = destination / rel / "index.md"
    with mkdocs_gen_files.open(page, "w") as handle:
        handle.write(f"# `{node.path}`\n\n::: {node.path}\n")


def _documentable_members(node: GriffeNode) -> Iterable[GriffeNode]:
    """Yield child modules/packages that should receive dedicated pages."""
    members_attr = getattr(node, "members", None)
    if not isinstance(members_attr, Mapping):
        return ()
    members: list[GriffeNode] = []
    for member in members_attr.values():
        member_node = cast(GriffeNode, member)
        if bool(getattr(member_node, "is_package", False)) or bool(
            getattr(member_node, "is_module", False)
        ):
            members.append(member_node)
    return tuple(members)


def generate_api_reference(
    loader: GriffeLoader,
    packages: Sequence[str],
    *,
    destination: Path | None = None,
) -> None:
    """Generate MkDocs API reference pages for ``packages``."""
    target = destination if destination is not None else Path("api")
    _write_index(target)

    for package in packages:
        module = cast(GriffeNode, loader.load(package))
        _write_node(target, module)
        for member in _documentable_members(module):
            _write_node(target, member)


def main(packages: Sequence[str] | None = None) -> int:
    """Entry point for the MkDocs API generator."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    resolved_packages = list(packages or iter_packages())
    generate_api_reference(LOADER, resolved_packages)

    LOG.info(
        "Generated MkDocs API reference",
        extra={
            "status": "success",
            "package_count": len(resolved_packages),
            "destination": "api",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
