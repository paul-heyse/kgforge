#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src"
uvicorn kgfoundry.search_api.app:app --host 0.0.0.0 --port 8080 --reload
