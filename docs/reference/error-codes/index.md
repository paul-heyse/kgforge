---
title: Canonical Error Codes
summary: Stable error codes surfaced by tooling, documentation pipelines, and runtime components.
---

# Canonical Error Codes

The following table enumerates the canonical error codes emitted by the
documentation pipeline and supporting tooling. Each code is globally unique
and may appear in terminal output, log events, Problem Details payloads, and
monitoring dashboards. Additional domains (API, runtime services, etc.) will
extend this registry using the same naming scheme.

| Code | Title | Domain | Category | Severity | Summary |
| ---- | ----- | ------ | -------- | -------- | ------- |
| `KGF-DOC-ENV-001` | Tooling prerequisites missing | documentation | environment | error | One or more prerequisite executables were not found. Run `scripts/bootstrap.sh` or `uv sync --frozen` to install the complete toolchain. |
| `KGF-DOC-ENV-002` | DocFacts schema not found | documentation | environment | error | `docs/_build/schema_docfacts.json` is missing. Restore it from version control or rerun the schema export utility before regenerating docs. |
| `KGF-DOC-BLD-001` | Docstring builder failed | documentation | build | error | Managed docstring generation exited with a non-zero status. Review the preceding log output for module-specific errors. |
| `KGF-DOC-BLD-002` | Docformatter failed | documentation | format | error | Docformatter detected formatting issues. Run the tool locally to view the file-level diff. |
| `KGF-DOC-BLD-003` | Pydocstyle validation failed | documentation | lint | error | Pydocstyle reported missing or malformed docstrings. The command output lists offending symbols. |
| `KGF-DOC-BLD-004` | Docstring coverage threshold not met | documentation | quality | error | Docstring coverage fell below the configured threshold. Add docstrings or update the policy if intentional. |
| `KGF-DOC-BLD-005` | DocFacts schema synchronization failed | documentation | build | error | Copying the canonical schema to `docs/_build/schema_docfacts.json` failed. Ensure `schema/docs/schema_docfacts.json` exists and rerun the stage. |
| `KGF-DOC-BLD-006` | DocFacts schema validation failed | documentation | validation | error | Docstring builder output violated the DocFacts schema. Synchronize the schema and rerun the docstring builder to regenerate artifacts. |
| `KGF-DOC-BLD-009` | Python compilation failed | documentation | build | error | Compiling sources with `python -m compileall -q src` failed. Run the command locally to locate and fix syntax errors before rebuilding docs. |
| `KGF-DOC-BLD-012` | Gallery validation failed | documentation | validation | error | `tools/validate_gallery.py` detected inconsistencies. Run the script with `--verbose` to inspect failing examples. |
| `KGF-DOC-BLD-013` | Example doctest suite failed | documentation | tests | error | Gallery doctests did not pass. Review the pytest output and update the affected examples. |
| `KGF-DOC-BLD-015` | README generation failed | documentation | build | error | Automated README generation or doctoc execution failed. Run `uv run python tools/gen_readmes.py` manually for diagnostics. |
| `KGF-DOC-BLD-010` | Navmap generation failed | documentation | build | error | Navigation map regeneration failed. Run `uv run python tools/navmap/build_navmap.py` to rewrite the canonical artifact (use `--write` to direct the output elsewhere). |
| `KGF-DOC-BLD-011` | Navmap integrity check failed | documentation | validation | error | Navmap validation detected drift. Ensure navigation artifacts are regenerated and up to date. |
| `KGF-DOC-BLD-020` | Symbol index build failed | documentation | build | error | AutoAPI symbol index generation failed. Investigate import errors or missing modules. |
| `KGF-DOC-BLD-021` | Symbol delta build failed | documentation | build | error | Symbol delta computation failed. Ensure symbol index artifacts exist and rerun `docs/_scripts/symbol_delta.py`. |
| `KGF-DOC-BLD-030` | Test map build failed | documentation | build | error | Test map artifacts could not be generated. Run `uv run python tools/docs/build_test_map.py` (configure via `TESTMAP_*` environment variables). |
| `KGF-DOC-BLD-040` | Observability scan failed | documentation | validation | error | Observability configuration scan exited with errors. Verify observability manifests. |
| `KGF-DOC-BLD-050` | Schema export failed | documentation | build | error | Exporting schemas failed. Confirm schema definitions and associated models compile. |
| `KGF-DOC-BLD-060` | Graph build failed | documentation | build | error | Dependency graph generation timed out or crashed. Ensure graph tooling (pydeps, pyreverse) is installed. |
| `KGF-DOC-BLD-070` | Sphinx HTML build failed | documentation | build | error | The Sphinx HTML builder aborted due to warnings or errors. Resolve the reported issues before retrying. |
| `KGF-DOC-BLD-080` | Sphinx JSON build failed | documentation | build | error | The Sphinx JSON builder aborted due to warnings or errors. Resolve the reported issues before retrying. |
| `KGF-DOC-BLD-090` | MkDocs build failed | documentation | build | error | MkDocs encountered an error while generating the static site. Inspect MkDocs output for details. |
| `KGF-DOC-BLD-091` | MkDocs suite build failed | documentation | build | error | The MkDocs suite configuration failed to build. Run `uv run mkdocs build --config-file tools/mkdocs_suite/mkdocs.yml --strict` to reproduce and address configuration issues. |
| `KGF-DOC-BLD-100` | Agent catalog build failed | documentation | build | error | Agent catalog generation failed. Ensure docfacts, navmap, and schema artifacts are current. |
| `KGF-DOC-BLD-105` | Documentation artifact validation failed | documentation | validation | error | Artifact validation encountered errors. Review `docs/_scripts/validate_artifacts.py` output for remediation steps. |

## Using Error Codes

* Each automated script should surface the error code when aborting.
* Logs should include the `error_code` structured field so observability
  backends can group failures.
* CI pipelines can gate on specific codes to produce actionable diagnostics.
* Developers can reference this registry to identify the canonical remediation
  guidance for each failure mode.

## Future Domains

The repository-wide naming convention follows **`KGF-<domain>-<category>-NNN`**.
Upcoming additions will cover API request validation, runtime service errors,
and deployment/operations issues. When adding a new code, update this registry
and the corresponding automation to ensure parity across tooling.

