"""Pytest marker to declare module requirements with optional GPU gating."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pytest

from kgfoundry_common.gpu import GPU_CORE_MODULES

REQ_PATTERN = re.compile(
    r"^\s*(?P<name>[a-zA-Z0-9_\.]+)(?:>=(?P<version>[\w\.\-]+))?\s*$",
)

GPU_MODULE_HEADS: set[str] = {name.split(".", 1)[0] for name in GPU_CORE_MODULES}

GPU_MODULE_FULL: set[str] = {
    "kgfoundry.vectorstore_faiss.gpu",
    "kgfoundry.search_api.faiss_adapter",
    "kgfoundry_common.faiss",
}

GPU_SKIP_REASON = "GPU stack (extra 'gpu') not installed/available in this environment"


INVALID_ARG_MSG = "requires marker arguments must be strings"
INVALID_MODULES_MSG = "requires 'modules' kwarg must be a list or tuple"
INVALID_ENTRY_MSG = "requires 'modules' entries must be strings"
INVALID_SYNTAX_MSG = "Invalid @pytest.mark.requires entry"


def _parse_entry(entry: str) -> tuple[str, str | None]:
    """Parse a requires marker entry into module name and optional min version."""
    match = REQ_PATTERN.match(entry)
    if not match:
        detail = f"{INVALID_SYNTAX_MSG}: {entry!r}"
        raise pytest.UsageError(detail)
    return match.group("name"), match.group("version")


def _iter_requirements(mark: pytest.Mark) -> Iterable[tuple[str, str | None]]:
    """Yield module requirements declared on the marker."""
    for arg in mark.args:
        if not isinstance(arg, str):
            raise pytest.UsageError(INVALID_ARG_MSG)
        yield _parse_entry(arg)
    modules = mark.kwargs.get("modules")
    if modules is not None:
        if not isinstance(modules, (list, tuple)):
            raise pytest.UsageError(INVALID_MODULES_MSG)
        for entry in modules:
            if not isinstance(entry, str):
                raise pytest.UsageError(INVALID_ENTRY_MSG)
            yield _parse_entry(entry)


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker description."""
    config.addinivalue_line(
        "markers",
        "requires(*mods, modules=None): import-or-skip listed modules; accepts 'pkg>=ver' syntax",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply import-or-skip semantics and optional GPU gating."""
    has_gpu_stack = getattr(config, "_has_gpu_stack", False)
    for item in items:
        mark = item.get_closest_marker("requires")
        if not mark:
            continue

        requirements = list(_iter_requirements(mark))
        gpu_related = False

        for module, min_version in requirements:
            reason_suffix = f" >= {min_version}" if min_version else ""
            pytest.importorskip(
                module,
                minversion=min_version,
                reason=f"Requires {module}{reason_suffix}",
            )
            head = module.split(".", 1)[0]
            if module in GPU_MODULE_FULL or head in GPU_MODULE_HEADS:
                gpu_related = True

        if gpu_related:
            item.add_marker(pytest.mark.gpu)
            if not has_gpu_stack:
                item.add_marker(pytest.mark.skip(reason=GPU_SKIP_REASON))
