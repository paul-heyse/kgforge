
.PHONY: bootstrap run api e2e test fmt lint clean migrations mock fixture

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

bootstrap:
	python3 -m venv $(VENV)
	$(PIP) install -U pip wheel
	$(PIP) install -U -r requirements-dev.txt
	$(PRECOMMIT) install
	$(PY) -m kgforge.registry.migrate apply --db /data/catalog/catalog.duckdb --migrations registry/migrations

run: api
api:
	$(UVICORN) kgforge.search_api.app:app --app-dir src --host 0.0.0.0 --port 8080 --reload

e2e:
	$(PYTEST) -q -m e2e || $(PYTEST) -q

test:
	$(PYTEST) -q

fmt:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/black src tests

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/mypy src

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache dist build

mock:
	$(PY) -m tests.mock_servers.run_all

fixture:
	$(PY) -m kgforge.orchestration.fixture_flow
