
.PHONY: bootstrap run api e2e test fmt lint clean migrations mock fixture docstrings docfacts-diff readmes html json symbols watch navmap-build navmap-check doctest test-map obs-catalog schemas graphs docs-html docs-json

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit
PKG := $(shell python tools/detect_pkg.py)
PKGS := $(shell python tools/detect_pkg.py --all)
WATCH_PORT := $(if $(SPHINX_AUTOBUILD_PORT),$(SPHINX_AUTOBUILD_PORT),8000)
DOCSTRING_DIRS := src tools docs/_scripts
FMT_TARGETS := src tests tools docs/_scripts
LINT_TARGETS := $(FMT_TARGETS)

bootstrap:
	python3 -m venv $(VENV)
	$(PIP) install -U pip wheel
	$(PIP) install -U -e ".[dev,docs]"
	$(PIP) install -U pre-commit
	$(PRECOMMIT) install
	$(PY) -m kgfoundry.registry.migrate apply --db /data/catalog/catalog.duckdb --migrations registry/migrations

run: api
api:
	$(UVICORN) kgfoundry.search_api.app:app --app-dir src --host 0.0.0.0 --port 8080 --reload

e2e:
	$(PYTEST) -q -m e2e || $(PYTEST) -q

test:
	$(PYTEST) -q

fmt:
	$(VENV)/bin/ruff check --select I --fix $(FMT_TARGETS)
	$(VENV)/bin/ruff check --fix $(FMT_TARGETS)
	$(VENV)/bin/ruff format $(FMT_TARGETS)
	$(VENV)/bin/black $(FMT_TARGETS)

lint:
	$(VENV)/bin/ruff check --select I $(LINT_TARGETS)
	$(VENV)/bin/ruff check $(LINT_TARGETS)
	$(VENV)/bin/ruff format --check $(FMT_TARGETS)
	$(VENV)/bin/mypy src

lint-docs:
	uvx pydoclint --style numpy src
	uv run python -m tools.docstring_builder.cli check --diff
	uvx mypy --strict src
	@if [ "$(RUN_DOCS_TESTS)" = "1" ]; then \
		uv run pytest tests/docs; \
	else \
		echo "Skipping tests/docs (set RUN_DOCS_TESTS=1 to enable)"; \
	fi

clean:
	rm -rf .venv .mypy_cache .ruff_cache .pytest_cache dist build

mock:
	$(PY) -m tests.mock_servers.run_all

fixture:
	$(PY) -m kgfoundry.orchestration.fixture_flow

docstrings:
	$(PY) tools/generate_docstrings.py
	$(PY) tools/update_navmaps.py
	$(VENV)/bin/docformatter --wrap-summaries=100 --wrap-descriptions=100 -r -i $(DOCSTRING_DIRS) || true
	$(VENV)/bin/pydocstyle $(DOCSTRING_DIRS)
	$(VENV)/bin/interrogate -i src --fail-under 90

docfacts-diff:
	uv run --no-project --with libcst --with griffe python -m tools.docstring_builder.cli --diff-only --all

readmes:
	$(PY) tools/gen_readmes.py
	-which doctoc >/dev/null 2>&1 && doctoc src/$(PKG) || echo "Install doctoc to auto-update TOCs."

html:
	$(PY) -m sphinx -b html -w sphinx-warn.log docs docs/_build/html

json:
	rm -rf docs/_build/json
	$(PY) -m sphinx -b json docs docs/_build/json

symbols:
	$(PY) docs/_scripts/build_symbol_index.py

navmap-build:
	$(PY) tools/navmap/build_navmap.py

navmap-check:
	$(PY) tools/navmap/check_navmap.py

watch:
	PYTHONPATH=src $(PY) -m sphinx_autobuild --port $(WATCH_PORT) docs docs/_build/html

doctest:
	$(PYTEST) -q --xdoctest --xdoctest-options=ELLIPSIS,IGNORE_WHITESPACE,NORMALIZE_WHITESPACE --xdoctest-modules --xdoctest-glob='examples/*.py' examples

test-map:
	$(PY) tools/docs/build_test_map.py

obs-catalog:
	$(PY) tools/docs/scan_observability.py

schemas:
	$(PY) tools/docs/export_schemas.py

graphs:
	$(PY) tools/docs/build_graphs.py

docs-html:
	$(MAKE) html

docs-json:
	$(MAKE) json
