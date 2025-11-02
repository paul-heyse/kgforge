## ADDED Requirements
### Requirement: Secure Serialization
The system SHALL replace insecure pickle usage with signed allow-listed serialization or safer formats, documenting the threat model and providing automated tests.

#### Scenario: Signed payload validates successfully
- **GIVEN** a payload serialized via `safe_pickle.dump` with an allowed class and signing key
- **WHEN** `safe_pickle.load` reads it
- **THEN** the payload is returned after signature verification and class validation, with tests proving round-trip behavior

#### Scenario: Tampered payload rejected
- **GIVEN** a payload where the signature or class name is altered
- **WHEN** `safe_pickle.load` executes
- **THEN** it raises `UnsafeSerializationError` and emits structured Problem Details logging the reason

### Requirement: Subprocess & Network Safety
The system SHALL enforce timeouts, sanitized paths, and ContextVar propagation for subprocess and network operations.

#### Scenario: Subprocess timeout enforced
- **GIVEN** a CLI command invoking a long-running subprocess
- **WHEN** the configured timeout elapses
- **THEN** the helper raises `SubprocessTimeoutError`, logs the failure, increments metrics, and propagates the error cause

#### Scenario: Network client respects timeout
- **GIVEN** an HTTP request issued through `search_client`
- **WHEN** the server stalls beyond the configured timeout
- **THEN** a `SearchTimeoutError` is raised with Problem Details payload and retry logic honors idempotency rules

### Requirement: Configuration Validation via Environment
The system SHALL load configuration exclusively from environment variables, validating values and providing smoke tests for missing/invalid configuration.

#### Scenario: Missing env produces failure
- **GIVEN** required env variables (e.g., `KGFOUNDRY_SIGNING_KEY`) are absent
- **WHEN** `load_config()` executes
- **THEN** it raises `ConfigurationError`, logs structured details, and tests assert the behavior

#### Scenario: Invalid timeout rejected
- **GIVEN** an environment variable specifying a negative timeout
- **WHEN** configuration validation runs
- **THEN** a `ValidationError` is raised, and smoke tests confirm the error message guides remediation

