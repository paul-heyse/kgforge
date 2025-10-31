# Tools Hardening — Contributor Guide

This change delivers typed contracts, secure subprocess orchestration, structured observability, and schema-backed CLI outputs across the `tools/` suite. Use this guide alongside `proposal.md`, `design.md`, `tasks.md`, and the capability deltas under `specs/tools-suite/`.

## Quick Start Checklist
1. Confirm the execution environment matches the repo toolchain (`scripts/bootstrap.sh`).
2. Review:
   - `proposal.md` — scope, acceptance, rollout expectations
   - `design.md` — architecture phases, typed models, error taxonomy, schemas
   - `tasks.md` — ordered runbook for automation agents
   - `specs/tools-suite/spec.md` — normative requirements and scenarios
3. Inspect the canonical contracts:
   - `tools/docstring_builder/models.py`
   - `tools/_shared/logging.py`
   - `tools/_shared/proc.py`
   - `schema/tools/docstring_builder_cli.json`
4. Capture baseline diagnostics (attach to execution note):
   ```bash
   uv run ruff check tools
   uv run pyrefly check tools
   uv run mypy tools
   python tools/check_imports.py
   uv run pip-audit --strict
   ```
5. Familiarize yourself with generated artifacts by running the existing pipelines (optional but recommended):
   ```bash
   uv run python -m tools.docstring_builder.cli generate --json --dry-run
   uv run python -m tools.docs.build_agent_catalog --help
   ```

## Deliverables Snapshot
- Typed APIs replace `dict[str, Any]` payloads across builders and docs pipelines.
- Shared logging/proc utilities enforce structured logs, Problem Details, and subprocess hygiene.
- Versioned base CLI envelope schema adopted across tools (`schema/tools/cli_envelope.json`) with `--json` support.
- Logging unified via `kgfoundry_common.logging` for consistent JSON formatting and correlation IDs.
- JSON Schemas for CLI/doc artifacts validated in tests (`schema/tools/*.json`).
- Refactored docstring builder modules, docs generators, and navmap utilities with clear error taxonomy.
- New pytest suites covering schema round-trips, CLI failure modes, plugin regression cases.

## Acceptance Gates (run before submission)
```bash
uv run ruff format && uv run ruff check --fix
uv run pyrefly check && uv run mypy --config-file mypy.ini
uv run pytest -q tests/tools
python tools/check_imports.py
uv run pip-audit --strict
make artifacts && git diff --exit-code
openspec validate tools-hardening --strict
```

## Junior playbook: implementing a migration

Follow these steps for each target file listed in `tasks.md` Appendix A.

1) Prepare
- Create a feature branch and run the baseline diagnostics above.
- Open the target file and identify locations of `subprocess.run` and `print(...)`.

2) Replace subprocess usage
- Import `run_tool` from `tools._shared.proc`.
- Swap `subprocess.run([...])` with `run_tool([...], timeout=<default>)`.
- Check `result.returncode`; on non-zero and `check=True` cases, raise the module’s typed error.

3) Replace prints with structured logs
- Import `get_logger, with_fields` from `tools._shared.logging`.
- Initialize `logger = get_logger(__name__)`.
- Log with `logger.info/warning/error` and attach `extra` fields via `with_fields`.

4) Add optional `--json` (for CLIs)
- Add an argparse flag `--json` (boolean).
- On completion, build the base envelope (see design.md Base CLI Envelope) and validate it against `schema/tools/cli_envelope.json`.
- Write the JSON to stdout; return a non-zero exit code on failure.

5) Tests
- Add/extend tests under `tests/tools/**` as per `tasks.md` Appendix A test list.

6) Verify
- Re-run Acceptance Gates. If drift is introduced, fix and repeat.

See `tasks.md` Appendix A for file-by-file checklists and code references.

## Questions & Support
- Schema changes → coordinate with documentation tooling owners.
- Observability metrics/logging → see `design.md` Observability section.
- CLI compatibility → ensure Problem Details samples stay synchronized with `docs/examples` fixtures.

