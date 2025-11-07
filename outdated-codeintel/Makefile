# CodeIntel Makefile
#
# Provides convenient targets for CodeIntel development and operations.
# Can be run from the codeintel directory or from the repo root.

.PHONY: serve index langs docs help

# Detect repo root - always use git to find it, or fall back to parent of codeintel
REPO_ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || dirname $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST)))))

# Always use uv from repo root context
UV := uv
UV_RUN := $(UV) run
PY := $(UV_RUN) python
TOOLS_DIR := $(REPO_ROOT)/tools

help:
	@echo "CodeIntel Makefile Targets:"
	@echo "  make serve      - Start the CodeIntel MCP server"
	@echo "  make index      - Build or update the persistent symbol index"
	@echo "  make langs      - Generate the language manifest (build/languages.json)"
	@echo "  make docs       - Generate CodeIntel MCP tools documentation"

serve:
	cd $(REPO_ROOT) && $(PY) -m codeintel.cli mcp serve --repo $(REPO_ROOT)

index:
	cd $(REPO_ROOT) && $(PY) -m codeintel.cli index build --repo $(REPO_ROOT)

langs:
	cd $(REPO_ROOT) && $(PY) -m codeintel.build_languages

docs:
	cd $(REPO_ROOT) && $(PY) $(TOOLS_DIR)/mkdocs_suite/docs/_scripts/gen_codeintel_mcp_docs.py

