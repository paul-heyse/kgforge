#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (one directory up from this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
BIN="$VENV/bin"

if [[ ! -x "$BIN/python" ]]; then
  echo "error: missing virtual environment at .venv/; run 'make bootstrap' first." >&2
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
PY="$BIN/python"

run() {
  printf '\n→ %s\n' "$*" >&2
  "$@"
}

ensure_tools() {
  local missing_tools=()
  for tool in doq docformatter pydocstyle pydoclint interrogate; do
    if [[ ! -x "$BIN/$tool" ]]; then
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

if [[ ! -x "$BIN/mkdocs" ]]; then
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
if ! run "$PY" tools/validate_gallery.py; then
  echo "Gallery validation failed. Fix errors before building docs." >&2
  exit 1
fi
run make readmes
run "$PY" tools/update_navmaps.py
run "$PY" tools/navmap/build_navmap.py
run "$PY" tools/navmap/check_navmap.py
run "$BIN"/pytest -q --xdoctest --xdoctest-options=ELLIPSIS,IGNORE_WHITESPACE,NORMALIZE_WHITESPACE --xdoctest-modules --xdoctest-glob='examples/*.py' examples
run make symbols
run "$PY" tools/docs/build_test_map.py
run "$PY" tools/docs/scan_observability.py
run "$PY" tools/docs/export_schemas.py
run "$PY" tools/docs/build_graphs.py
run make html
run make json
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  run "$BIN/mkdocs" build --config-file mkdocs.yml
fi

printf '\nDocumentation refresh complete.\n' >&2
