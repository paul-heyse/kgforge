"""Generate folder-cluster D2 diagrams for MkDocs navigation."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Final

import mkdocs_gen_files

GROUP_DEPTH_ENV_VAR: Final = "MKDOCS_D2_GROUP_DEPTH"
DEFAULT_GROUP_DEPTH: Final = 1
DOCS_ROOT = Path(__file__).resolve().parents[1]
CURATED_INDEX_PATH = DOCS_ROOT / "diagrams" / "index.md"


def _load_curated_intro() -> str:
    """Return the curated diagrams landing page introduction.

    The static ``docs/diagrams/index.md`` file contains human-crafted context
    explaining how the generated diagrams are structured. We seed the generated
    index with that prose before appending the dynamic folder listing so that
    documentation builds retain the curated onboarding experience.
    """

    try:
        content = CURATED_INDEX_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "# Diagrams\n\n"

    if content.endswith("\n\n"):
        return content
    if content.endswith("\n"):
        return f"{content}\n"
    return f"{content}\n\n"


def _resolve_group_depth() -> int:
    """Return the configured module prefix depth used for grouping diagrams.

    Reads the ``MKDOCS_D2_GROUP_DEPTH`` environment variable to determine
    how many module path segments should be used for grouping. Defaults to
    1 if not set or if the value is invalid.

    Returns
    -------
    int
        Group depth (number of module prefix segments), always at least 1.
    """
    value = os.environ.get(GROUP_DEPTH_ENV_VAR)
    if value is None:
        return DEFAULT_GROUP_DEPTH

    try:
        depth = int(value)
    except ValueError:
        return DEFAULT_GROUP_DEPTH

    return max(1, depth)


def main() -> None:
    """Entry point executed by mkdocs-gen-files."""
    group_depth = _resolve_group_depth()
    by_folder: dict[str, list[str]] = defaultdict(list)
    for file in list(mkdocs_gen_files.files):
        path = file.src_uri
        if not path.startswith("modules/") or not path.endswith(".md"):
            continue
        module_path = path[len("modules/") : -3]
        if module_path == "index":
            continue
        parts = module_path.split(".")
        depth = min(len(parts), group_depth)
        folder = ".".join(parts[:depth])
        by_folder[folder].append(module_path)

    for folder in sorted(by_folder):
        modules = by_folder[folder]
        d2_path = f"diagrams/{folder}.d2"
        with mkdocs_gen_files.open(d2_path, "w") as handle:
            handle.write("direction: right\n")
            handle.write(f'{folder}: "{folder}" {{\n')
            for module_path in sorted(modules):
                handle.write(
                    f'  "{module_path}": "{module_path}" {{ link: "../modules/{module_path}.md" }}\n'
                )
            handle.write("}\n")

    curated_intro = _load_curated_intro()

    with mkdocs_gen_files.open("diagrams/index.md", "w") as handle:
        handle.write(curated_intro)
        if not curated_intro.endswith("\n\n"):
            handle.write("\n")
        for folder in sorted(by_folder):
            handle.write(f"- [{folder}](./{folder}.d2)\n")


main()
