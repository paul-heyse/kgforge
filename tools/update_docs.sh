#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (one directory up from this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "error: missing virtual environment at .venv/; run 'make bootstrap' first." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-src}"
export SPHINXOPTS="-W"

run() {
  printf '\n→ %s\n' "$*" >&2
  "$@"
}

# Ensure we rebuild from a clean slate.
rm -rf docs/_build/html docs/_build/json site

run make docstrings
run make readmes
run make html
run make json
run make symbols
run .venv/bin/mkdocs build --config-file mkdocs.yml

printf '\nDocumentation refresh complete.\n' >&2
