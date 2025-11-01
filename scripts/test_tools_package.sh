#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST_DIR="${ROOT_DIR}/dist"

echo "[tools-package] Building wheel in ${DIST_DIR}" >&2
uv run --extra tools python -m build --wheel --outdir "${DIST_DIR}"

LATEST_WHEEL=$(ls -t "${DIST_DIR}"/kgfoundry-*.whl | head -n 1)
if [[ -z "${LATEST_WHEEL}" ]]; then
  echo "No kgfoundry wheel found in ${DIST_DIR}" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

python -m venv "${TMP_DIR}/venv"
source "${TMP_DIR}/venv/bin/activate"
python -m pip install --upgrade pip setuptools
python -m pip install "${LATEST_WHEEL}[tools]"

python - <<'PY'
import tools

result = tools.run_tool(["python", "--version"], check=False)
print("tools.run_tool executed successfully:", result.returncode)
PY

deactivate
echo "[tools-package] Smoke test completed successfully" >&2

