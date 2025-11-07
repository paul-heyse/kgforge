# CodeIntel Resource Limits

CodeIntel enforces configurable resource limits to prevent resource exhaustion and ensure safe operation.

## Environment Variables

All limits are configurable via environment variables with sensible defaults.

### Size Limits

- **`CODEINTEL_MAX_AST_BYTES`** (default: `1048576` = 1 MiB)
  - Maximum file size for AST extraction
  - Files exceeding this limit return Problem Details error
  - Prevents memory exhaustion on large files

- **`CODEINTEL_MAX_OUTLINE_ITEMS`** (default: `2000`)
  - Maximum number of outline items per file
  - Prevents unbounded outline generation

### List Limits

- **`CODEINTEL_LIMIT_DEFAULT`** (default: `100`)
  - Default limit for list operations (`code.listFiles`, etc.)
  - Applied when `limit` parameter is not specified

- **`CODEINTEL_LIMIT_MAX`** (default: `1000`)
  - Hard maximum limit for all list operations
  - User-specified limits are clamped to this value

### Time Limits

- **`CODEINTEL_TOOL_TIMEOUT_S`** (default: `10.0`)
  - Maximum execution time per tool call (seconds)
  - Tools exceeding this timeout return Problem Details (504)
  - Uses AnyIO cancellation scopes for clean timeout handling

### Rate Limiting

- **`CODEINTEL_RATE_LIMIT_QPS`** (default: `5.0`)
  - Tokens per second refill rate
  - Controls sustained request rate

- **`CODEINTEL_RATE_LIMIT_BURST`** (default: `10`)
  - Maximum token capacity (burst size)
  - Allows short bursts above QPS rate

Rate limiting uses a token bucket algorithm. Requests exceeding the rate limit return Problem Details (429).

### Feature Flags

- **`CODEINTEL_ENABLE_TS_QUERY`** (default: `"0"` = disabled)
  - Enable advanced Tree-sitter query tool (`ts.query`)
  - Set to `"1"` to enable
  - Disabled by default for security (prevents arbitrary query execution)

## Examples

### Increase AST Size Limit

```bash
export CODEINTEL_MAX_AST_BYTES=5242880  # 5 MiB
python -m codeintel.cli mcp serve --repo .
```

### Allow Higher Request Rate

```bash
export CODEINTEL_RATE_LIMIT_QPS=20.0
export CODEINTEL_RATE_LIMIT_BURST=50
python -m codeintel.cli mcp serve --repo .
```

### Enable Advanced Queries

```bash
export CODEINTEL_ENABLE_TS_QUERY=1
python -m codeintel.cli mcp serve --repo .
```

### Longer Timeouts for Large Repos

```bash
export CODEINTEL_TOOL_TIMEOUT_S=30.0
python -m codeintel.cli mcp serve --repo .
```

## Problem Details

When limits are exceeded, CodeIntel returns RFC 9457 Problem Details:

- **429 (Too Many Requests):** Rate limit exceeded
- **504 (Gateway Timeout):** Tool execution timeout
- **400 (Bad Request):** File size exceeds limit
- **403 (Forbidden):** Advanced query disabled

All errors include:
- `type`: `urn:kgf:problem:codeintel:*`
- `code`: Machine-readable error code (e.g., `KGF-CI-RATE`)
- `detail`: Human-readable explanation
- `extensions`: Additional context (limits, timeouts, etc.)

