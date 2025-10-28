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
  local missing=0
  for tool in doq docformatter pydocstyle pydoclint interrogate; do
    if [[ ! -x "$BIN/$tool" ]]; then
      echo "error: missing '$tool'; install docs extras via 'pip install -e \".[docs]\"' (inside .venv)." >&2
      missing=1
    fi
  done
  if [[ $missing -ne 0 ]]; then
    exit 1
  fi
}

ensure_tools

if [[ ! -x "$BIN/mkdocs" ]]; then
  echo "warning: mkdocs not installed; skipping mkdocs build (install via 'pip install -e \".[docs-mkdocs]\"')." >&2
  BUILD_MKDOCS=0
else
  BUILD_MKDOCS=1
fi

# Ensure we rebuild from a clean slate.
rm -rf docs/_build site

run make docstrings
run make readmes
run "$PY" tools/update_navmaps.py
run "$PY" tools/navmap/build_navmap.py
run "$PY" tools/navmap/check_navmap.py
run "$BIN"/pytest -q --xdoctest --xdoctest-options=ELLIPSIS,IGNORE_WHITESPACE,NORMALIZE_WHITESPACE --xdoctest-modules --xdoctest-glob='examples/*.py' examples
run "$PY" tools/docs/build_test_map.py
run "$PY" tools/docs/scan_observability.py
run "$PY" tools/docs/export_schemas.py
run "$PY" tools/docs/build_graphs.py
run make html
run make json
run make symbols
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  run "$BIN/mkdocs" build --config-file mkdocs.yml
fi

printf '\nDocumentation refresh complete.\n' >&2
