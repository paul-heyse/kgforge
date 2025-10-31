# Src Hardening Phase 2 — Agent Catalog & Search APIs

Phase 2 tackles the highest volume Ruff/mypy/pyrefly violations in the agent catalog and search services (FAISS/SPLADE/BM25/fixture indexes, FastAPI app, registry adapters). This guide complements `proposal.md`, `design.md`, `tasks.md`, and the capability deltas under `specs/src-agent-search/`.

## Quick Start Checklist
1. Run `scripts/bootstrap.sh` to ensure toolchain parity (Python 3.13.9 + `uv`).
2. Read:
   - `proposal.md` — scope, impact, acceptance
   - `design.md` — architecture plan, typed APIs, schema strategy
   - `tasks.md` — ordered runbook + appendices
   - `specs/src-agent-search/spec.md` — normative requirements and scenarios
   - `openspec/changes/src-hardening-phase-1/` outputs (Phase 1 foundations)
3. Inspect canonical modules:
   - `src/kgfoundry/agent_catalog/search.py`, `session.py`, `cli.py`, `mcp.py`, `audit.py`
   - `src/search_api/app.py`, `service.py`, `faiss_adapter.py`, `bm25_index.py`, `splade_index.py`, `fixture_index.py`
   - `src/registry/*.py` for DuckDB integrations
   - `tests/search_api/**` and data fixtures
4. Capture baseline diagnostics (attach to execution note):
   ```bash
   uv run ruff check src/kgfoundry/agent_catalog src/search_api
   uv run pyrefly check src/kgfoundry/agent_catalog src/search_api
   uv run mypy src/kgfoundry/agent_catalog src/search_api
   python tools/check_imports.py
   uv run pip-audit --strict
   ```
5. Collect representative sample payloads for API/CLI responses (for schema validation tests).

## Deliverables Snapshot
- Typed FAISS/SPLADE/BM25 interfaces via Protocols and TypedDicts; removal of `Any` flows.
- Secure subprocess & SQL execution (parameterized queries, sanitized inputs) across search and registry modules.
- Search API returns RFC 9457 Problem Details with schema validation; structured logs & metrics across failure paths.
- CLI and MCP interfaces emit versioned JSON envelopes aligning with `schema/tools/cli_envelope.json` + new agent catalog schemas.
- Expanded pytest suites covering search edge cases, plugin regression, SQL injection attempts, and error paths.
- Updated documentation (schemas, API reference, migration notes) and optional extras for dependencies (FAISS, DuckDB, numpy).

## Acceptance Gates (run before submission)
```bash
uv run ruff format && uv run ruff check --fix src/kgfoundry/agent_catalog src/search_api src/registry
uv run pyrefly check src/kgfoundry/agent_catalog src/search_api
uv run mypy src/kgfoundry/agent_catalog src/search_api src/registry
uv run pytest -q tests/search_api tests/agent_catalog
python tools/check_imports.py
uv run pip-audit --strict
make artifacts && git diff --exit-code
openspec validate src-hardening-phase-2 --strict
```

## Questions & Support
- FAISS/SPLADE APIs → coordinate with ML platform owners.
- SQL/registry changes → sync with data platform team for DuckDB guidelines.
- API contracts → align with API guild for Problem Details envelope & schema additions.

Agents must log progress via the execution note template referenced in `tasks.md`.

