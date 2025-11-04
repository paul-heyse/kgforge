
.PHONY: bootstrap run api e2e test test-gpu test-cpu fmt lint lint-gpu-gates clean migrations mock fixture docstrings docfacts-diff readmes html json symbols watch navmap-build navmap-check doctest test-map obs-catalog schemas graphs docs-html docs-json artifacts build_agent_catalog

UV := uv
UV_RUN := $(UV) run
UVX := uvx
PY := $(UV_RUN) python
PIP := $(UV_RUN) pip
UVICORN := $(UV_RUN) uvicorn
PYTEST := $(UV_RUN) pytest
PRECOMMIT := $(UV_RUN) pre-commit
PKG := $(shell $(UV_RUN) python tools/detect_pkg.py)
PKGS := $(shell $(UV_RUN) python tools/detect_pkg.py --all)
WATCH_PORT := $(if $(SPHINX_AUTOBUILD_PORT),$(SPHINX_AUTOBUILD_PORT),8000)
DOCSTRING_DIRS := src tools docs/_scripts
FMT_TARGETS := src tests tools docs/_scripts
LINT_TARGETS := $(FMT_TARGETS)

bootstrap:
	bash scripts/bootstrap.sh

run: api
api:
	$(UVICORN) kgfoundry.search_api.app:app --app-dir src --host 0.0.0.0 --port 8080 --reload

e2e:
	$(PYTEST) -q -m e2e || $(PYTEST) -q

test:
	$(PYTEST) -q

test-gpu:
	$(PYTEST) -m gpu -q

test-cpu:
	$(PYTEST) -m "not gpu" -q

fmt:
	$(UV_RUN) ruff check --select I --fix $(FMT_TARGETS)
	$(UV_RUN) ruff check --fix $(FMT_TARGETS)
	$(UV_RUN) ruff format $(FMT_TARGETS)
	$(UV_RUN) black $(FMT_TARGETS)

lint:
	$(UV_RUN) ruff check --select I $(LINT_TARGETS)
	$(UV_RUN) ruff check $(LINT_TARGETS)
	$(UV_RUN) ruff format --check $(FMT_TARGETS)
	$(UV_RUN) mypy src

lint-gpu-gates:
	$(PY) tools/lint/check_gpu_marks.py

lint-docs:
	$(UVX) pydoclint --style numpy src
	$(UV_RUN) python -m tools.docstring_builder.cli check --diff
	$(UVX) mypy --strict src
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
	$(UV_RUN) docformatter --wrap-summaries=100 --wrap-descriptions=100 -r -i $(DOCSTRING_DIRS) || true
	$(UV_RUN) pydocstyle $(DOCSTRING_DIRS)
	$(UV_RUN) docstr-coverage --fail-under 90 src

docfacts-diff:
	uv run --no-project --with libcst --with griffe python -m tools.docstring_builder.cli --diff-only --all

stubs-check:
	uv run --no-project --with libcst --with griffe --with mkdocs-gen-files python tools/stubs/drift_check.py

artifacts:
	uv run python tools/docs/build_artifacts.py

build_agent_catalog:
	uv run python tools/docs/build_agent_catalog.py
	uv run python tools/docs/build_agent_api.py
	uv run python tools/docs/render_agent_portal.py
	uv run python tools/docs/build_agent_analytics.py

readmes:
	$(PY) tools/gen_readmes.py
	@if $(UV) run --which doctoc >/dev/null 2>&1; then \
		$(UV_RUN) doctoc src/$(PKG); \
	else \
		echo "Install doctoc to auto-update TOCs."; \
	fi

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

uv-pytest:
	uv run --env PYTHONPATH=src pytest

uv-test:
	uv run --env PYTHONPATH=src pytest -q

make-test:
	PYTHONPATH=src $(PYTEST) -q
