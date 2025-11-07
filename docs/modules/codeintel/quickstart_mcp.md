# CodeIntel MCP Quickstart

This guide walks you through setting up CodeIntel MCP server and connecting it to ChatGPT.

## Prerequisites

- Python 3.13.9 (managed via `uv`)
- Dependencies installed (`scripts/bootstrap.sh`)
- Tree-sitter languages built (`make codeintel-langs`)

## Step 1: Build Language Grammars

Ensure Tree-sitter grammars are available:

```bash
make codeintel-langs
# or
python -m codeintel.build_languages
```

This generates `codeintel/build/languages.json` with installed grammar versions.

## Step 2: Start the Server

### Option A: Direct Server

```bash
python -m codeintel.mcp_server.server
```

### Option B: Via CLI Façade (Recommended)

```bash
python -m codeintel.cli mcp serve --repo .
```

The CLI façade provides:
- Correlation IDs for request tracking
- Structured envelope output
- Environment variable management

## Step 3: Connect to ChatGPT

1. Open ChatGPT Settings → **Capabilities / MCP**
2. Choose **Add local server**
3. Configure:
   - **Command:** `python -m codeintel.mcp_server.server`
   - **Working directory:** `/path/to/your/repo` (absolute path)
   - **Environment variables (optional):**
     ```
     CODEINTEL_MAX_AST_BYTES=1048576
     CODEINTEL_RATE_LIMIT_QPS=5
     CODEINTEL_RATE_LIMIT_BURST=10
     CODEINTEL_ENABLE_TS_QUERY=1
     ```

ChatGPT should handshake and display available tools.

## Step 4: First Calls

Try these MCP tool calls:

### List Files

```json
{
  "method": "tools/call",
  "params": {
    "name": "code.listFiles",
    "arguments": {
      "directory": ".",
      "glob": "src/**/*.py",
      "limit": 10
    }
  }
}
```

### Get Outline

```json
{
  "method": "tools/call",
  "params": {
    "name": "code.getOutline",
    "arguments": {
      "path": "src/kgfoundry/search.py",
      "language": "python"
    }
  }
}
```

### Get AST

```json
{
  "method": "tools/call",
  "params": {
    "name": "code.getAST",
    "arguments": {
      "path": "src/kgfoundry/search.py",
      "language": "python",
      "format": "json"
    }
  }
}
```

### Tree-sitter Query (if enabled)

```json
{
  "method": "tools/call",
  "params": {
    "name": "ts.query",
    "arguments": {
      "path": "src/kgfoundry/search.py",
      "language": "python",
      "query": "(function_definition name: (identifier) @def.name)"
    }
  }
}
```

## Troubleshooting

### No Response

- Ensure JSON is newline-terminated
- Check server is reading from stdin
- Verify working directory matches repo root

### Sandbox Errors

- Check `KGF_REPO_ROOT` environment variable
- Verify paths are repository-relative
- Ensure no `../` traversal attempts

### Large File Errors

- Increase `CODEINTEL_MAX_AST_BYTES`
- Use `code.getFile` with offset/length for chunked reads

### Rate Limit Errors

- Increase `CODEINTEL_RATE_LIMIT_QPS` and `CODEINTEL_RATE_LIMIT_BURST`
- Reduce request frequency

### TS Query Disabled

- Set `CODEINTEL_ENABLE_TS_QUERY=1` to enable advanced queries
- Default is disabled for security

## Next Steps

- Read [Tools Reference](./tools.md) for all available tools
- See [Limits](./limits.md) for resource cap configuration
- Check [Config](./config.md) for environment variable details

