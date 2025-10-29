## ADDED Requirements

### Requirement: Data Model Helper Extended Summaries
The system SHALL supply multi-sentence extended summaries for docstrings generated from data-model helpers, transport protocols, and exception plumbing so numpydoc ES01 validation passes without manual intervention.

#### Scenario: Exception chaining helpers documented
- **WHEN** `tools/generate_docstrings.py` processes `__cause__`, `__context__`, or other exception-chaining attributes
- **THEN** the generated docstrings include extended summaries that explain how Python populates the attribute, when it changes, and how developers should use it while handling errors

#### Scenario: Metadata containers receive context
- **WHEN** the docstring generator encounters classes like `ModuleMeta`, `_NoopMetric`, or other containers created by documentation tooling
- **THEN** the extended summary clarifies their purpose in the build pipeline (e.g., storing module metadata, acting as a sentinel) and why they appear in public API docs

#### Scenario: Data loading helpers described
- **WHEN** helper functions such as `_load_dense_from_parquet` or `from_parquet` are regenerated
- **THEN** their extended summaries describe the file format, expected schema, and common usage pattern within the toolkit

#### Scenario: Protocol helper types documented
- **WHEN** the generator emits docstrings for structural protocols like `SupportsHttp` or `SupportsResponse`
- **THEN** extended summaries explain which methods the protocol expects, how they integrate with HTTP clients, and when to implement them

#### Scenario: Fallback text still passes ES01
- **WHEN** a new helper class or function matches the detection logic but lacks an explicit template
- **THEN** the fallback extended summary still contains at least two informative sentences, closing with a period, and passes numpydoc ES01 validation

