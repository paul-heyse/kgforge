
.PHONY: bootstrap run api e2e test fmt lint clean migrations mock fixture docstrings readmes html json symbols watch

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit
PKG := $(shell python tools/detect_pkg.py)
PKGS := $(shell python tools/detect_pkg.py --all)
WATCH_PORT := $(if $(SPHINX_AUTOBUILD_PORT),$(SPHINX_AUTOBUILD_PORT),8000)

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

docstrings:
	@for pkg in $(PKGS); do \
		echo "Updating docstrings for $$pkg"; \
		$(VENV)/bin/doq --formatter google -t tools/doq_templates/google -w -r -d src/$$pkg; \
	done
	$(VENV)/bin/docformatter -r -i src
	$(VENV)/bin/pydocstyle src
	$(VENV)/bin/interrogate -i src --fail-under 90

readmes:
	$(PY) tools/gen_readmes.py
	-which doctoc >/dev/null 2>&1 && doctoc src/$(PKG) || echo "Install doctoc to auto-update TOCs."

html:
	$(PY) -m sphinx -b html docs docs/_build/html

json:
	rm -rf docs/_build/json
	$(PY) -m sphinx -b json docs docs/_build/json

symbols:
	$(PY) docs/_scripts/build_symbol_index.py

watch:
	PYTHONPATH=src $(PY) -m sphinx_autobuild --port $(WATCH_PORT) docs docs/_build/html
