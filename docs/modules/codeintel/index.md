# CodeIntel (Tree-sitter + MCP)

CodeIntel exposes your repository structure to MCP clients (e.g., ChatGPT) via a local stdio server. It's safe-by-default (sandboxed), fast, and agent-friendly.

- **Server:** `python -m codeintel.mcp_server.server`
- **CLI:** `python -m codeintel.cli mcp serve --repo .`
- **Tools:** outline, AST, TS query, file list/get (see [tools.md](./tools.md))
- **Limits:** size/time/rate caps (see [limits.md](./limits.md))
- **Config:** environment variables and repository root (see [config.md](./config.md))

## Quick Start

1. **Start the server:**
   ```bash
   python -m codeintel.cli mcp serve --repo .
   ```

2. **Connect from ChatGPT:**
   - Add a local MCP server
   - Command: `python -m codeintel.mcp_server.server`
   - Working directory: your repo root

3. **Try it:**
   - `code.listFiles` — list repository files
   - `code.getOutline` — get file structure
   - `code.getAST` — get syntax tree

See [Quickstart Guide](./quickstart_mcp.md) for detailed setup instructions.

## Features

- **Multi-language support:** Python, JSON, YAML, TOML, Markdown
- **Safe sandboxing:** All paths validated against repository root
- **Resource limits:** Configurable size, time, and rate limits
- **Persistent index:** Optional SQLite index for fast symbol search
- **Problem Details:** RFC 9457 compliant error responses

## Architecture

- **Language Runtime:** Manifest-driven Tree-sitter grammar loading
- **MCP Server:** JSON-RPC 2.0 stdio server with rate limiting and timeouts
- **Tools:** Sandboxed file operations with size caps
- **Index:** Optional SQLite store for symbols and references

