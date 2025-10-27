# Getting Started

This guide walks you through the documentation workflow for kgfoundry.

1. Install the documentation extras once: `pip install -e ".[docs]"`.
2. Generate docstrings and normalize formatting with `make docstrings`.
3. Refresh package READMEs with deep links via `make readmes` (optionally set `DOCS_LINK_MODE` before running).
4. Build the docs corpus with `make html`, `make json`, and `make symbols` as needed.
5. Run `make watch` for live reloading while editing content or code.

Export `DOCS_LINK_MODE=github` when you need GitHub permalinks instead of local editor links. Override `DOCS_EDITOR` to switch between VS Code and PyCharm deep links.
