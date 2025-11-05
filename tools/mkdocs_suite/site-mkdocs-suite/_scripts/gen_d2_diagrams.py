"""Generate folder-cluster D2 diagrams for MkDocs navigation."""

from __future__ import annotations

from collections import defaultdict

import mkdocs_gen_files


def main() -> None:
    """Entry point executed by mkdocs-gen-files."""
    by_folder: dict[str, list[str]] = defaultdict(list)
    for file in list(mkdocs_gen_files.files):
        path = file.src_uri
        if not path.startswith("modules/") or not path.endswith(".md"):
            continue
        module_path = path[len("modules/") : -3]
        if module_path == "index":
            continue
        parts = module_path.split(".")
        folder = parts[1] if len(parts) > 1 else parts[0]
        by_folder[folder].append(module_path)

    for folder, modules in by_folder.items():
        d2_path = f"diagrams/{folder}.d2"
        with mkdocs_gen_files.open(d2_path, "w") as handle:
            handle.write("direction: right\n")
            handle.write(f'{folder}: "{folder}" {{\n')
            for module_path in sorted(modules):
                handle.write(
                    f'  "{module_path}": "{module_path}" {{ link: "../modules/{module_path}.md" }}\n'
                )
            handle.write("}\n")

    with mkdocs_gen_files.open("diagrams/index.md", "w") as handle:
        handle.write("# Diagrams\n\n")
        for folder in sorted(by_folder):
            handle.write(f"- [{folder}](./{folder}.d2)\n")


main()
