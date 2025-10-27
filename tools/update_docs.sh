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

export PYTHONPATH="${PYTHONPATH:-src}"
export SPHINXOPTS="-W"
PY="$BIN/python"

run() {
  printf '\n→ %s\n' "$*" >&2
  "$@"
}

ensure_tools() {
  local missing=0
  for tool in doq docformatter pydocstyle interrogate; do
    if [[ ! -x "$BIN/$tool" ]]; then
      echo "error: missing '$tool'; install docs extras via 'pip install -e \".[docs,docs-agent]\"' (inside .venv)." >&2
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
rm -rf docs/_build/html docs/_build/json site

run make docstrings
run make readmes
run make html
run make json
run make symbols
if [[ $BUILD_MKDOCS -eq 1 ]]; then
  run "$BIN/mkdocs" build --config-file mkdocs.yml
fi

printf '\nDocumentation refresh complete.\n' >&2
