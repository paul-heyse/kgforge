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

run() {
  printf '\n→ %s\n' "$*" >&2
  "$@"
}

uv_run() {
  run uv run "$@"
}

ensure_tools() {
  local missing_tools=()
  for tool in doq docformatter pydocstyle pydoclint docstr-coverage; do
    if ! uv run --which "$tool" >/dev/null 2>&1; then
      missing_tools+=("$tool")
    fi
  done
  if ((${#missing_tools[@]} != 0)); then
    echo "error: missing required tools: ${missing_tools[*]}." >&2
    echo "       Run 'uv sync --frozen' (or './scripts/bootstrap.sh')." >&2
    exit 1
  fi
}

ensure_tools

if ! uv run --which mkdocs >/dev/null 2>&1; then
  echo "error: missing 'mkdocs'; run 'uv sync --frozen' (or './scripts/bootstrap.sh')." >&2
  exit 1
fi
BUILD_MKDOCS=1

# Ensure we rebuild from a clean slate.
rm -rf docs/_build site

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

run make docstrings
if ! uv_run python tools/validate_gallery.py; then
  echo "Gallery validation failed. Fix errors before building docs." >&2
  exit 1
fi

run make readmes
uv_run python tools/update_navmaps.py
uv_run python tools/navmap/build_navmap.py
uv_run python tools/navmap/check_navmap.py
uv_run pytest -q --xdoctest --xdoctest-options=ELLIPSIS,IGNORE_WHITESPACE,NORMALIZE_WHITESPACE --xdoctest-modules --xdoctest-glob='examples/*.py' examples
run make symbols
uv_run python tools/docs/build_test_map.py
uv_run python tools/docs/scan_observability.py
uv_run python tools/docs/export_schemas.py
uv_run python tools/docs/build_graphs.py
run make html
run make json
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  uv_run mkdocs build --config-file mkdocs.yml
fi
run make build_agent_catalog

printf '\nDocumentation refresh complete.\n' >&2
