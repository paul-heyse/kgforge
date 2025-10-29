"""Overview of mkdocs gen api.

This module bundles mkdocs gen api logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mkdocs_gen_files

if TYPE_CHECKING:
    from griffe import Object as GriffeObject
else:
    GriffeObject = object  # type: ignore[misc, assignment]

griffe_module = import_module("griffe")
Object = cast(Any, griffe_module.Object)
try:
    loader_module = import_module("griffe.loader")
except ModuleNotFoundError:
    GriffeLoader = cast(Any, griffe_module.GriffeLoader)
else:
    GriffeLoader = cast(Any, loader_module.GriffeLoader)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS_DIR = ROOT / "tools"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.detect_pkg import detect_packages, detect_primary  # noqa: E402

out = Path("api")
with mkdocs_gen_files.open(out / "index.md", "w") as f:
    f.write("# API Reference\n")


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Returns
    -------
    List[str]
        Description of return value.

    Examples
    --------
    >>> from docs._scripts.mkdocs_gen_api import iter_packages
    >>> result = iter_packages()
    >>> result  # doctest: +ELLIPSIS
    """
    packages = detect_packages()
    return packages or [detect_primary()]


loader = GriffeLoader(search_paths=[str(SRC if SRC.exists() else ROOT)])


def write_node(node: GriffeObject) -> None:
    """Compute write node.

    Carry out the write node operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    node : typing.Any
        Description for ``node``.

    Examples
    --------
    >>> from docs._scripts.mkdocs_gen_api import write_node
    >>> write_node(...)  # doctest: +ELLIPSIS
    """
    rel = node.path.replace(".", "/")
    page = out / rel / "index.md"
    with mkdocs_gen_files.open(page, "w") as f:
        f.write(f"# `{node.path}`\n\n::: {node.path}\n")


for pkg in iter_packages():
    module = loader.load(pkg)
    write_node(module)
    for member in module.members.values():
        if member.is_package or member.is_module:
            write_node(member)
