#!/usr/bin/env bash
# Bootstrap a reproducible dev env (base + docs only) using uv.
# Works on Linux and macOS (Bash 3.2+ compatible; avoids arrays/process substitution).
#
# Behavior:
# - Creates .venv if missing
# - Installs project editable + extras=docs (NOT gpu)
# - Respects uv.lock (frozen); optionally offline
# - Activates .venv for this shell
# - Installs pre-commit hooks
# - Provides helpful diagnostics and exits nonzero on failure

set -euo pipefail

# -------- pretty prints --------
is_tty() { [ -t 1 ]; }
color()  { is_tty && command -v tput >/dev/null && tput setaf "$1" || true; }
reset()  { is_tty && command -v tput >/dev/null && tput sgr0 || true; }
log()    { printf "%s%s%s\n" "$(color 4)" "$*" "$(reset)"; }
ok()     { printf "%s%s%s\n" "$(color 2)" "$*" "$(reset)"; }
warn()   { printf "%s%s%s\n" "$(color 3)" "WARN: $*" "$(reset)"; }
err()    { printf "%s%s%s\n" "$(color 1)" "ERROR: $*" "$(reset)"; }

# -------- repo root detection --------
# Resolve to repo root even when invoked via symlink
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------- configurable knobs (env overrides allowed) --------
PYTHON_SPEC="${PYTHON_SPEC:-3.13}"
EXTRAS="${EXTRAS:-docs}"           # NEVER include "gpu" here
FROZEN="${FROZEN:-1}"               # 1 = respect uv.lock; 0 = allow relock/updates
OFFLINE="${OFFLINE:-0}"             # 1 = no network (uses cache / local wheelhouse)
USE_WHEELHOUSE="${USE_WHEELHOUSE:-0}" # 1 = add ./.wheelhouse to candidate wheels

# Hard safety: block gpu extra unless explicitly allowed
ALLOW_GPU="${ALLOW_GPU:-0}"
case ",${EXTRAS}," in
  *,gpu,*) if [ "${ALLOW_GPU}" != "1" ]; then
              err "The 'gpu' extra is disallowed by default. Set ALLOW_GPU=1 to proceed (not recommended for bootstrap)."
              exit 2
            fi
            ;;
esac

# -------- prerequisites --------
need_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }; }
need_cmd uv
log "uv $(uv --version || echo "(version unknown)")"

# Optional: pin Python (writes .python-version) if missing
if [ ! -f ".python-version" ]; then
  log "Pinning Python ${PYTHON_SPEC} → .python-version"
  uv python pin "${PYTHON_SPEC}" || warn "Could not pin Python; proceeding."
fi

# -------- venv creation --------
if [ ! -d ".venv" ]; then
  log "Creating .venv with uv"
  uv venv
else
  log ".venv already exists"
fi

# -------- compose sync flags --------
SYNC_FLAGS=()
[ "${FROZEN}" = "1" ] && SYNC_FLAGS+=(--frozen)
[ "${OFFLINE}" = "1" ] && SYNC_FLAGS+=(--offline)
IFS=',' read -r EX1 EX2 EX3 EX4 EX5 <<<"${EXTRAS},,,,"
for E in "${EX1}" "${EX2}" "${EX3}" "${EX4}" "${EX5}"; do
  [ -n "${E}" ] && SYNC_FLAGS+=(--extra "${E}")
done

# Add local wheelhouse (flat index) if requested
if [ "${USE_WHEELHOUSE}" = "1" ] && [ -d ".wheelhouse" ]; then
  export UV_FIND_LINKS="${UV_FIND_LINKS:-./.wheelhouse}"
  log "Using local wheelhouse: ${UV_FIND_LINKS}"
fi

# -------- sync env (editable project + extras) --------
log "Syncing environment: uv sync ${SYNC_FLAGS[*]}"
uv sync ${SYNC_FLAGS[@]}

# -------- activate venv for this shell --------
ACTIVATE=".venv/bin/activate"
if [ -f "${ACTIVATE}" ]; then
  # shellcheck disable=SC1090
  . "${ACTIVATE}"
  ok "Activated .venv for this shell"
else
  warn "Could not auto-activate .venv; use:  . .venv/bin/activate"
fi

# -------- install pre-commit hooks --------
if command -v uvx >/dev/null 2>&1; then
  log "Installing pre-commit hooks"
  uvx pre-commit install -t pre-commit -t pre-push || warn "pre-commit installation skipped"
else
  warn "uvx not found; skipping pre-commit hook install"
fi

# -------- sanity: packaging & scripts --------
if grep -qE '^\[project\.scripts\]' pyproject.toml 2>/dev/null; then
  if ! grep -qE '^\[build-system\]' pyproject.toml 2>/dev/null && \
     ! grep -qE '^\[tool\.uv\][[:space:]]*$' pyproject.toml 2>/dev/null && \
     ! grep -qE '^package[[:space:]]*=' pyproject.toml 2>/dev/null; then
    warn "You defined [project.scripts] but no build backend / tool.uv.package. Consider adding a build backend."
  fi
fi

# -------- quick tips --------
ok "Environment ready (base + docs, no gpu). Next steps:"
echo "  - Lint/format : uvx ruff check --fix && uvx ruff format"
echo "  - Type-check  : uvx mypy --strict"
echo "  - Tests       : uv run pytest -q"
echo "  - Build docs  : uv run sphinx-build -b html docs docs/_build/html"
