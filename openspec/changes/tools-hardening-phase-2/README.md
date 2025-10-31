# tools-hardening-phase-2

- [proposal](./proposal.md)
- [tasks](./tasks.md)
- [spec delta](./specs/tools-suite/spec.md)

### Validation Commands
- `uv run ruff format && uv run ruff check --fix`
- `uv run pyrefly check`
- `uv run mypy --config-file mypy.ini`
- `uv run pip-audit --strict`
- `python tools/check_imports.py`
- `spectral lint path/to/openapi.yaml` *(when HTTP surfaces exist)*
- `uv run pytest -q`
- `make artifacts`


