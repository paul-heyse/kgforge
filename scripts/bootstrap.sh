#!/usr/bin/env bash
# Bootstrap a reproducible dev env (standard deps only) using uv.
# Works on Linux and macOS (Bash 3.2+ compatible; avoids arrays/process substitution).
#
# Behavior:
# - Creates .venv if missing
# - Installs project editable with the standard dependency set (no gpu extras)
# - Respects uv.lock (frozen); optionally offline
# - Activates .venv for this shell
# - Installs pre-commit hooks
# - Provides helpful diagnostics and exits nonzero on failure

set -euo pipefail

REQUIRED_UV_VERSION="${REQUIRED_UV_VERSION:-0.93.0}"
REQUIRED_PYTHON_VERSION="${REQUIRED_PYTHON_VERSION:-3.13.9}"

# Prefer uv-managed Python (avoid picking system Python)
export UV_MANAGED_PYTHON=true

# -------- pretty prints --------
is_tty() { [ -t 1 ]; }
color()  { is_tty && command -v tput >/dev/null && tput setaf "$1" || true; }
reset()  { is_tty && command -v tput >/dev/null && tput sgr0 || true; }
log()    { printf "%s%s%s\n" "$(color 4)" "$*" "$(reset)"; }
ok()     { printf "%s%s%s\n" "$(color 2)" "$*" "$(reset)"; }
warn()   { printf "%s%s%s\n" "$(color 3)" "WARN: $*" "$(reset)"; }
err()    { printf "%s%s%s\n" "$(color 1)" "ERROR: $*" "$(reset)"; }

version_ge() {
  local lhs="${1:-}"
  local rhs="${2:-}"
  if [ -z "${lhs}" ] || [ -z "${rhs}" ]; then
    return 1
  fi
  [ "$(printf '%s\n%s\n' "${lhs}" "${rhs}" | sort -V | tail -n1)" = "${lhs}" ]
}

version_gt() {
  local lhs="${1:-}"
  local rhs="${2:-}"
  version_ge "${lhs}" "${rhs}" && [ "${lhs}" != "${rhs}" ]
}

ensure_uv_installed() {
  local required_version="${1}"
  local current_version=""
  local needs_install=1
  if command -v uv >/dev/null 2>&1; then
    current_version="$(uv --version 2>/dev/null | awk '{print $2}')"
    if [ -n "${current_version}" ] && version_gt "${current_version}" "${required_version}"; then
      needs_install=0
    fi
  fi
  if [ "${needs_install}" -eq 1 ]; then
    log "Installing uv (requires version > ${required_version})"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    hash -r
  else
    log "uv ${current_version} already satisfies > ${required_version}"
  fi
}

ensure_uv_python_version() {
  local version="${1:-}"
  if [ -z "${version}" ]; then
    err "Missing REQUIRED_PYTHON_VERSION"
    exit 1
  fi
  if ! uv python list "${version}" --only-installed | grep -q "${version}"; then
    log "Installing Python ${version} via uv"
    uv python install "${version}"
  else
    log "uv-managed Python ${version} already installed"
  fi
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 1; }; }

ensure_uv_installed "${REQUIRED_UV_VERSION}"
need_cmd uv
log "uv $(uv --version || echo "(version unknown)")"
ensure_uv_python_version "${REQUIRED_PYTHON_VERSION}"

# -------- repo root detection --------
# Resolve to repo root even when invoked via symlink
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -------- configurable knobs (env overrides allowed) --------
REQUIRED_PYTHON_VERSION="${REQUIRED_PYTHON_VERSION:-3.13.9}"
PYTHON_SPEC="${PYTHON_SPEC:-${REQUIRED_PYTHON_VERSION}}"
EXTRAS="${EXTRAS:-}"                # Provide comma-separated extras as needed (never include "gpu")
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
  log "Upgrading pip tooling via uv"
  uv pip install -U pip certifi
  log "Installing cuvs-cu13 from NVIDIA index via uv"
  uv pip install --extra-index-url https://pypi.nvidia.com cuvs-cu13
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
ok "Environment ready (standard deps, no gpu). Next steps:"
echo "  - Lint/format : uvx ruff check --fix && uvx ruff format"
echo "  - Type-check  : uvx mypy --strict"
echo "  - Tests       : uv run pytest -q"
echo "  - Build docs  : uv run sphinx-build -b html docs docs/_build/html"
