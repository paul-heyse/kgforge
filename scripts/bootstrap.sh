#!/usr/bin/env bash
set -euo pipefail

python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[search,docs]"

echo "Apply DuckDB migrations with your local duckdb client if needed."
echo "Configure systemd units under config/systemd/ and nginx with config/nginx.conf."
