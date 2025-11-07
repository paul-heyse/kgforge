# ADR 001: Exception Taxonomy with RFC 9457 Problem Details

**Status**: Accepted  
**Date**: 2024-11-07  
**Context**: CodeIntel Enhancement Implementation

## Context

CodeIntel requires a robust, production-ready error handling strategy that:
1. Provides programmatic distinction between error types
2. Supplies machine-readable error details for clients
3. Aligns with web API standards for HTTP boundaries
4. Maintains consistency with the rest of KGFoundry

## Decision

We implemented a **custom exception hierarchy** inheriting from `kgfoundry_common.errors.KGFError`, with full RFC 9457 Problem Details support.

### Exception Types Created

```python
CodeIntelError (base)
├── SandboxError (403)
├── LanguageNotSupportedError (400)
├── QuerySyntaxError (422)
├── IndexNotFoundError (404)
├── FileTooLargeError (413)
├── ManifestError (500)
├── OperationTimeoutError (504)
├── RateLimitExceededError (429)
└── IndexCorruptedError (500)
```

### Key Design Principles

1. **Inheritance from KGFError**: Ensures consistent error handling across KGFoundry
2. **RFC 9457 Compliance**: Each exception includes:
   - `problem_type` - URN identifying the error category
   - `default_status` - HTTP status code
   - `default_title` - Human-readable summary
   - `extensions` - Arbitrary context data
3. **No String Parsing**: Clients can programmatically handle errors by type
4. **Context Preservation**: `raise ... from e` maintains cause chains

### Example Usage

```python
from codeintel.errors import LanguageNotSupportedError

raise LanguageNotSupportedError(
    "Language 'rust' not supported",
    extensions={"requested": "rust", "available": ["python", "json"]}
)
```

Generates Problem Details:
```json
{
  "type": "urn:kgf:problem:codeintel:language-not-supported",
  "title": "Language not supported",
  "status": 400,
  "detail": "Language 'rust' not supported",
  "requested": "rust",
  "available": ["python", "json"]
}
```

## Consequences

### Positive

- ✅ **Type-safe error handling**: `except LanguageNotSupportedError` instead of string matching
- ✅ **Client-friendly**: Machine-readable errors with structured context
- ✅ **HTTP-ready**: Direct mapping to status codes for HTTP boundaries
- ✅ **Debuggable**: Cause chains preserved with `from e`
- ✅ **Extensible**: Easy to add new exception types

### Negative

- ⚠️ **More code**: 10 exception classes vs generic `ValueError`
- ⚠️ **Learning curve**: Team must learn exception hierarchy

### Neutral

- 📊 **Consistency**: Aligns with KGFoundry standards (`KGFError` base)
- 📊 **Migration**: Old code needs updating to use new exceptions

## Alternatives Considered

### 1. Generic Exceptions (ValueError, RuntimeError)
**Rejected**: No programmatic distinction, poor client experience

### 2. Error Codes (integer codes)
**Rejected**: Requires documentation lookup, not Pythonic

### 3. Dataclasses for Errors
**Rejected**: Not compatible with Python exception machinery

## References

- RFC 9457: https://www.rfc-editor.org/rfc/rfc9457.html
- `kgfoundry_common.errors.KGFError`: Base exception class
- AGENTS.md Principle 1: "Exceptions are part of the contract"

