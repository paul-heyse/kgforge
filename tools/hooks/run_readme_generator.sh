#!/usr/bin/env bash
set -euo pipefail

# Determine which packages have staged changes under src/ and regenerate the
# corresponding READMEs.  Developers can override the package selection by
# exporting DOCS_PKG before invoking the hook.

if [[ -n "${DOCS_PKG:-}" ]]; then
  packages="$DOCS_PKG"
else
  mapfile -t changed < <(git diff --cached --name-only --relative -- 'src/**')
  declare -A unique_pkgs=()
  for path in "${changed[@]}"; do
    [[ -z "$path" ]] && continue
    pkg=${path#src/}
    pkg=${pkg%%/*}
    [[ -z "$pkg" ]] && continue
    unique_pkgs[$pkg]=1
  done
  if [[ ${#unique_pkgs[@]} -eq 0 ]]; then
    echo "readme-generator: no tracked package changes detected; skipping." >&2
    exit 0
  fi
  packages=$(IFS=,; echo "${!unique_pkgs[*]}")
fi

export DOCS_PKG="$packages"

echo "readme-generator: regenerating READMEs for packages: $packages" >&2
python tools/gen_readmes.py --link-mode github --editor relative

# Stage updated READMEs so the hook behaves like other formatting hooks.
git add src/**/README.md
