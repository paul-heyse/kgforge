## Context
- `kgfoundry_common.serialization` still relies on `pickle` without signature checks or allow-lists, exposing potential code execution risks.
- Multiple subprocess calls (`orchestration/cli.py`, doc tooling) lack explicit timeouts and path sanitization; async contexts do not propagate ContextVars.
- Network clients (`search_client/client.py`) call `requests` without timeouts, increasing risk of hanging connections.
- Configuration loading mixes hard-coded defaults and environment reads, lacking validation on missing/invalid values.
- Documentation does not summarize threat model or configuration expectations related to serialization and subprocess/network usage.

## Goals / Non-Goals
- **Goals**
  - Replace or wrap pickle usage with signed payload checks or safer serialization formats.
  - Enforce timeouts, sanitized paths, and ContextVar propagation on subprocess and network operations.
  - Ensure configuration relies on environment variables with strict validation and smoke tests for failure cases.
  - Document the security threat model and configuration guidance.
- **Non-Goals**
  - Introducing new authentication protocols or complex cryptography beyond signing/validation needs.
  - Overhauling entire orchestration workflows beyond security/timeouts.
  - Replacing third-party libraries (e.g., `requests`) unless necessary.

## Decisions
- Implement `safe_pickle` module with allow-listed classes and HMAC signature verification; migrate existing pickle usage to this module.
- Provide JSON-based alternative serialization for new payloads, reserving pickle only when necessary and documented.
- For subprocess calls, use `subprocess.run(..., timeout=seconds, check=True)` with sanitized `Path` arguments and environment via `env` dict derived from settings.
- Introduce `AsyncRequestContext` using `contextvars` to propagate correlation IDs/timeouts across async tasks.
- For network operations, add explicit timeouts (`timeout=(connect_timeout, read_timeout)`), retries where appropriate, and tests verifying behavior.
- Expand `AppSettings` (Phase 3 outcome) to include security-related configuration (allowed hosts, timeout defaults) with validated fields.
- Add tests verifying configuration fails fast when required env vars missing or invalid (e.g., negative timeouts).
- Document threat model: trusted/untrusted payload boundaries, signature management, recommended timeout defaults.

## Alternatives
- Use `pickle` with static allow-list only — rejected; signature verification adds additional protection.
- Rely solely on manual audits for subprocess inputs — rejected; automated sanitization and tests needed.
- Use environment variables parsed manually — rejected; rely on `pydantic_settings` for validation.

## Risks / Trade-offs
- Signing payloads introduces key management overhead.
  - Mitigation: Store signing keys in config/env; document rotation process; add tests ensuring signature validation.
- Enforcing timeouts may cause previously successful long-running operations to fail.
  - Mitigation: Provide configurable timeout values with documented defaults.
- Additional validation may impact performance slightly.
  - Mitigation: Profile changes and keep signature verification efficient; only apply to relevant payloads.

## Migration
- Introduce `safe_pickle` wrapper with compatibility mode; update consumers gradually.
- Add configuration defaults and environment variable documentation; provide sample `.env` snippet.
- Update tests to cover failure paths before enabling strict validation.
- Document migration steps in release notes, including instructions for enabling signed payloads and setting timeouts.

