#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
PYVER=${PYVER:-"3.13"}                  # or 3.12
CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-13.0"}
ARCH_LIST=${ARCH_LIST:-"9.0"}          # RTX 50xx (Blackwell)
XFORMERS_REF=${XFORMERS_REF:-"main"}    # or a release tag, e.g., v0.0.32.post2
PRETEND_VER=${PRETEND_VER:-""}          # e.g., 0.0.33+cu130 to match strict pins

# === ENV ===
uv venv .xformer-build-venv --python "$PYVER"
source .xformer-build-venv/bin/activate
uv pip install wheel build setuptools "setuptools-scm>=8" ninja cmake

uv pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu130

# === FETCH ===
rm -rf xformers
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout "$XFORMERS_REF"

# === BUILD ===
export CUDA_HOME="$CUDA_HOME"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="$(nproc)"
export FORCE_CUDA=1
if [[ -n "$PRETEND_VER" ]]; then
  export SETUPTOOLS_SCM_PRETEND_VERSION="$PRETEND_VER"
fi

uv build --wheel --no-isolation
echo "Wheel built at: $(ls -1 dist/xformers-*.whl)"
#python -m pip install dist/xformers-*.whl
#python -m xformers.info
