#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (one directory up from this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: missing 'uv'; run './scripts/bootstrap.sh' first." >&2
  exit 1
fi

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT/src:$ROOT"
else
  case ":$PYTHONPATH:" in
    *":$ROOT/src:"*) ;;
    *) PYTHONPATH="$PYTHONPATH:$ROOT/src" ;;
  esac
  case ":$PYTHONPATH:" in
    *":$ROOT:"*) ;;
    *) PYTHONPATH="$PYTHONPATH:$ROOT" ;;
  esac
  export PYTHONPATH
fi
export SPHINXOPTS="-W"

PACKAGE_NAME="$(uv run python tools/detect_pkg.py)"
PACKAGE_VERSION="$(
  uv run python - "$PACKAGE_NAME" <<'PY'
from __future__ import annotations

import importlib.metadata as metadata
import sys

package = sys.argv[1]

try:
    print(metadata.version(package))
except metadata.PackageNotFoundError:
    print("0.0.0")
PY
)"

emit_error_message() {
  KGF_ERROR_CODE="$1" KGF_ERROR_MESSAGE="$2" KGF_ERROR_DETAILS="${3:-}" \
    uv run python - <<'PY'
import os
from tools._shared.error_codes import format_error_message

code = os.environ["KGF_ERROR_CODE"]
message = os.environ["KGF_ERROR_MESSAGE"]
details = os.environ.get("KGF_ERROR_DETAILS") or None
print(format_error_message(code, message, details=details))
PY
}

announce_stage() {
  local code="$1"
  local description="$2"
  printf '\n[START %s] %s\n' "$code" "$description"
}

complete_stage() {
  local code="$1"
  printf '[OK %s] Completed successfully.\n' "$code"
}

run() {
  printf '\n→ %s\n' "$*" >&2
  "$@"
}

uv_run() {
  run uv run "$@"
}

fail_with_code() {
  local code="$1"; shift
  local message="$1"; shift
  local details="$*"
  emit_error_message "$code" "$message" "$details" >&2
  printf 'See docs/reference/error-codes/index.md for remediation guidance.\n' >&2
  exit 1
}

ensure_tools() {
  local missing_tools=()
  for tool in doq docformatter pydocstyle pydoclint docstr-coverage; do
    if ! uv run which "$tool" >/dev/null 2>&1; then
      missing_tools+=("$tool")
    fi
  done
  if ((${#missing_tools[@]} != 0)); then
    fail_with_code "KGF-DOC-ENV-001" "Missing documentation tooling prerequisites" \
      "Missing executables: ${missing_tools[*]}"
  fi
}

ensure_tools

if ! uv run which mkdocs >/dev/null 2>&1; then
  fail_with_code "KGF-DOC-ENV-001" "MkDocs executable not found" \
    "Install mkdocs via 'uv sync --frozen' or rerun scripts/bootstrap.sh."
fi
BUILD_MKDOCS=1

SCHEMA_DOCFACTS_PATH="docs/_build/schema_docfacts.json"

SCHEMA_DOCFACTS_CANONICAL="schema/docs/schema_docfacts.json"

# Ensure we rebuild from a clean slate.
rm -rf docs/_build site site-mkdocs-suite
mkdir -p docs/_build

# keep graph small
export GRAPH_NOISE_LEVEL=50
export GRAPH_MAX_MODULE_DEPTH=1
export GRAPH_MAX_BACON=4

# cycle safety
export GRAPH_CYCLE_LEN=8          # only cycles up to length 8
export GRAPH_CYCLE_LIMIT=50000    # never enumerate more than 50k cycles
export GRAPH_EDGE_BUDGET=50000    # if pruned graph has >50k edges, skip enumeration and summarize SCCs

export GRAPH_MAX_WORKERS=16
export GRAPH_FAIL_ON_CYCLES=0
export GRAPH_FAIL_ON_LAYER=0

announce_stage "KGF-DOC-BLD-005" "Synchronizing DocFacts schema"
KGF_ROOT="$ROOT" uv_run python - <<'PY'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

root = Path(os.environ["KGF_ROOT"])
source = root / "schema" / "docs" / "schema_docfacts.json"
target = root / "docs" / "_build" / "schema_docfacts.json"

if not source.exists():
    print(f"MISSING_CANONICAL_SCHEMA:{source}", file=sys.stderr)
    sys.exit(2)

target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, target)
PY
status=$?
if ((status != 0)); then
  if ((status == 2)); then
    fail_with_code "KGF-DOC-ENV-002" "DocFacts schema not found" \
      "Expected canonical schema at $SCHEMA_DOCFACTS_CANONICAL."
  else
    fail_with_code "KGF-DOC-BLD-005" "DocFacts schema synchronization failed"
  fi
fi
complete_stage "KGF-DOC-BLD-005"

DOCSTRING_DIRS=(src tools docs/_scripts)

announce_stage "KGF-DOC-BLD-001" "Generating managed docstrings"
if ! uv_run python tools/generate_docstrings.py; then
  fail_with_code "KGF-DOC-BLD-001" "Docstring builder failed"
fi
complete_stage "KGF-DOC-BLD-001"

announce_stage "KGF-DOC-BLD-002" "Formatting docstrings"
if ! uv_run docformatter --wrap-summaries=100 --wrap-descriptions=100 -r -i "${DOCSTRING_DIRS[@]}"; then
  fail_with_code "KGF-DOC-BLD-002" "Docformatter exited with a non-zero status"
fi
complete_stage "KGF-DOC-BLD-002"

announce_stage "KGF-DOC-BLD-003" "Validating docstring style"
if ! uv_run pydocstyle "${DOCSTRING_DIRS[@]}"; then
  fail_with_code "KGF-DOC-BLD-003" "pydocstyle reported violations"
fi
complete_stage "KGF-DOC-BLD-003"

announce_stage "KGF-DOC-BLD-004" "Checking docstring coverage"
if ! uv_run docstr-coverage --fail-under 90 src; then
  fail_with_code "KGF-DOC-BLD-004" "Docstring coverage threshold not met"
fi
complete_stage "KGF-DOC-BLD-004"

announce_stage "KGF-DOC-BLD-016" "Generating CLI OpenAPI specification"
if ! uv_run python tools/typer_to_openapi_cli.py \
  --app orchestration.cli:app \
  --bin kgf \
  --title "KgFoundry CLI" \
  --version "$PACKAGE_VERSION" \
  --augment openapi/_augment_cli.yaml \
  --interface-id orchestration-cli \
  --out openapi/openapi-cli.yaml; then
  fail_with_code "KGF-DOC-BLD-016" "CLI OpenAPI generation failed"
fi
if ! cp openapi/openapi-cli.yaml tools/mkdocs_suite/docs/openapi/openapi-cli.yaml; then
  fail_with_code "KGF-DOC-BLD-016" "Failed to publish CLI OpenAPI specification" \
    "Ensure tools/mkdocs_suite/docs/openapi/ exists and is writable."
fi
complete_stage "KGF-DOC-BLD-016"

announce_stage "KGF-DOC-BLD-012" "Validating gallery references"
if ! uv_run python tools/validate_gallery.py; then
  fail_with_code "KGF-DOC-BLD-012" "Gallery validation failed"
fi
complete_stage "KGF-DOC-BLD-012"

announce_stage "KGF-DOC-BLD-015" "Generating README content"
if ! uv_run python tools/gen_readmes.py; then
  fail_with_code "KGF-DOC-BLD-015" "README generation failed"
fi
if [[ -n "$PACKAGE_NAME" && -d "src/$PACKAGE_NAME" ]] && uv run which doctoc >/dev/null 2>&1; then
  if ! uv_run doctoc "src/$PACKAGE_NAME"; then
    fail_with_code "KGF-DOC-BLD-015" "Doctoc execution failed for README content"
  fi
fi
complete_stage "KGF-DOC-BLD-015"

announce_stage "KGF-DOC-BLD-009" "Compiling Python sources"
if ! uv_run python -m compileall -q src; then
  fail_with_code "KGF-DOC-BLD-009" "Python source compilation failed"
fi
complete_stage "KGF-DOC-BLD-009"

announce_stage "KGF-DOC-BLD-010" "Updating navigation map"
if ! uv_run python tools/update_navmaps.py; then
  fail_with_code "KGF-DOC-BLD-010" "Navigation map update failed"
fi
if ! uv_run python tools/navmap/build_navmap.py; then
  fail_with_code "KGF-DOC-BLD-010" "Navmap build failed"
fi
if ! uv_run python tools/navmap/check_navmap.py; then
  fail_with_code "KGF-DOC-BLD-011" "Navmap integrity check failed"
fi
complete_stage "KGF-DOC-BLD-011"

announce_stage "KGF-DOC-BLD-013" "Running gallery doctests"
if ! uv_run pytest -q --xdoctest --xdoctest-options=ELLIPSIS,IGNORE_WHITESPACE,NORMALIZE_WHITESPACE --xdoctest-modules --xdoctest-glob='examples/*.py' examples; then
  fail_with_code "KGF-DOC-BLD-013" "Example doctest suite failed"
fi
complete_stage "KGF-DOC-BLD-013"

announce_stage "KGF-DOC-BLD-020" "Generating symbol index"
if ! uv_run python docs/_scripts/build_symbol_index.py; then
  fail_with_code "KGF-DOC-BLD-020" "Symbol index build failed"
fi
complete_stage "KGF-DOC-BLD-020"

announce_stage "KGF-DOC-BLD-030" "Building test map artifacts"
if ! uv_run python tools/docs/build_test_map.py; then
  fail_with_code "KGF-DOC-BLD-030" "Test map build failed"
fi
complete_stage "KGF-DOC-BLD-030"

announce_stage "KGF-DOC-BLD-040" "Scanning observability configuration"
if ! uv_run python tools/docs/scan_observability.py; then
  fail_with_code "KGF-DOC-BLD-040" "Observability scan failed"
fi
complete_stage "KGF-DOC-BLD-040"

announce_stage "KGF-DOC-BLD-050" "Exporting schemas"
if ! uv_run python tools/docs/export_schemas.py; then
  fail_with_code "KGF-DOC-BLD-050" "Schema export failed"
fi
complete_stage "KGF-DOC-BLD-050"

announce_stage "KGF-DOC-BLD-105" "Validating documentation artifacts"
if ! uv_run python docs/_scripts/validate_artifacts.py; then
  fail_with_code "KGF-DOC-BLD-105" "Documentation artifact validation failed"
fi
complete_stage "KGF-DOC-BLD-105"

announce_stage "KGF-DOC-BLD-060" "Building dependency graphs"
if ! uv_run python tools/docs/build_graphs.py; then
  fail_with_code "KGF-DOC-BLD-060" "Graph build failed"
fi
complete_stage "KGF-DOC-BLD-060"

announce_stage "KGF-DOC-BLD-070" "Building Sphinx HTML documentation"
if ! uv_run python -m sphinx -b html -w sphinx-warn.log docs docs/_build/html; then
  fail_with_code "KGF-DOC-BLD-070" "Sphinx HTML build failed"
fi
complete_stage "KGF-DOC-BLD-070"

announce_stage "KGF-DOC-BLD-080" "Building Sphinx JSON documentation"
rm -rf docs/_build/json
if ! uv_run python -m sphinx -b json docs docs/_build/json; then
  fail_with_code "KGF-DOC-BLD-080" "Sphinx JSON build failed"
fi
complete_stage "KGF-DOC-BLD-080"

announce_stage "KGF-DOC-BLD-090" "Building MkDocs site"
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  if ! uv_run mkdocs build --config-file mkdocs.yml; then
    fail_with_code "KGF-DOC-BLD-090" "MkDocs build failed"
  fi
fi
complete_stage "KGF-DOC-BLD-090"

announce_stage "KGF-DOC-BLD-091" "Building MkDocs suite site"
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  if ! uv_run mkdocs build --config-file tools/mkdocs_suite/mkdocs.yml --site-dir site-mkdocs-suite; then
    fail_with_code "KGF-DOC-BLD-091" "MkDocs suite build failed"
  fi
fi
complete_stage "KGF-DOC-BLD-091"

printf '\nDocumentation refresh complete.\n' >&2
