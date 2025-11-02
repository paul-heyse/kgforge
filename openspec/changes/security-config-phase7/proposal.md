## Why
Legacy serialization paths still use raw `pickle`, several subprocess/network calls run without explicit timeouts or sanitized paths, and configuration loading lacks thorough validation. These gaps expose security risks and violate 12-factor principles.

## What Changes
- [x] **MODIFIED**: Serialization layers replace insecure pickle usage with signed payload checks or migrating to safer formats; threat model documented.
- [x] **MODIFIED**: Subprocess and network operations updated to enforce timeouts, sanitize paths, and propagate ContextVars in async contexts.
- [x] **MODIFIED**: Configuration loading audited to rely solely on environment variables, with smoke tests covering missing/invalid configurations and failure-on-invalid behavior.
- [x] **ADDED**: Security documentation describing mitigation steps, threat model assumptions, and validation procedures.
- [ ] **REMOVED**: Deprecated helpers that bypass security controls.

## Impact
- **Affected specs (capabilities):** `security-config/core`
- **Affected code paths:** `src/kgfoundry_common/serialization.py`, `src/orchestration/cli.py`, `src/kgfoundry_common/fs.py`, network clients (`search_client/client.py`), configuration modules, tests verifying security behavior
- **Data contracts:** None directly, but signed payloads and configuration schemas must be documented
- **Rollout plan:** Implement serialization changes alongside deprecation of old code paths; stage rollout with feature flags if necessary; communicate new configuration requirements.

## Acceptance
- [ ] All serialization pathways use secure formats or signed allow-listed pickle wrappers; threat model documented.
- [ ] Subprocess/network calls enforce timeouts, sanitized inputs, and propagate ContextVars; tests cover failure behavior.
- [ ] Configuration loading relies on environment variables, fails fast on missing/invalid values, and includes smoke tests demonstrating error handling.
- [ ] Security documentation updated; Ruff, Pyrefly, Mypy, pytest remain green.

## Out of Scope
- Implementing new authentication/authorization flows.
- Network protocol redesigns.
- Large-scale dependency upgrades.

## Risks / Mitigations
- **Risk:** Replacing serialization may break backward compatibility.
  - **Mitigation:** Provide migration helpers, maintain compatibility mode for existing payloads, and add tests for legacy data.
- **Risk:** Enforcing timeouts may surface latent bugs.
  - **Mitigation:** Document new defaults, offer configuration overrides, and add tests confirming behavior.
- **Risk:** Strict configuration validation might break existing deployments.
  - **Mitigation:** Provide clear migration guide and defaults for new settings; add smoke tests to illustrate expected usage.

## Alternatives Considered
- Leaving pickle unchanged with warnings — rejected due to security risk.
- Adding timeouts only in CLI — rejected; network clients and services also require protection.

