# CodeIntel Configuration

Configuration for CodeIntel MCP server via environment variables and repository structure.

## Repository Root

The server operates within a repository sandbox defined by `KGF_REPO_ROOT`.

### Environment Variable

- **`KGF_REPO_ROOT`** (default: current working directory)
  - Absolute path to repository root
  - All file operations are validated against this root
  - Set automatically by CLI façade: `python -m codeintel.cli mcp serve --repo .`

### Path Resolution

- All paths are resolved relative to `KGF_REPO_ROOT`
- Paths outside the repository root are rejected with `SandboxError`
- Symlink traversal is prevented via `Path.resolve()` checks

## Excluded Patterns

The following patterns are excluded from file operations:

- `**/.git/**`
- `**/.venv/**`
- `**/_build/**`
- `**/__pycache__/**`
- `**/.mypy_cache/**`
- `**/.pytest_cache/**`
- `**/node_modules/**`

These exclusions apply to:
- `code.listFiles` directory scans
- Persistent index building (`codeintel index build`)

## Language Support

Supported languages (via Tree-sitter):

- **Python** (`python`) — Full support with queries
- **JSON** (`json`) — Basic support
- **YAML** (`yaml`) — Basic support
- **TOML** (`toml`) — Basic support
- **Markdown** (`markdown`) — Basic support

Language detection is based on file extension:
- `.py` → `python`
- `.json` → `json`
- `.yaml`, `.yml` → `yaml`
- `.toml` → `toml`
- `.md` → `markdown`

## Query Files

Tree-sitter queries are loaded from `codeintel/queries/{language}.scm`:

- `python.scm` — Function definitions and calls
- `toml.scm` — Top-level keys
- `yaml.scm` — First-level mappings
- `markdown.scm` — Headings and code blocks

Missing query files result in empty outlines (no error).

## Persistent Index

Optional SQLite index stored at `.kgf/codeintel.db`:

- **Schema:** `codeintel/index/schema.sql`
- **Build:** `python -m codeintel.cli index build --repo .`
- **Fresh rebuild:** `python -m codeintel.cli index build --repo . --fresh`

Index tracks:
- File metadata (mtime, size)
- Symbol definitions (functions, classes)
- Reference edges (calls, imports)

Incremental indexing skips unchanged files (mtime + size check).

## Advanced Query Feature

The `ts.query` tool is **disabled by default** for security:

- Requires `CODEINTEL_ENABLE_TS_QUERY=1` to enable
- Allows arbitrary Tree-sitter S-expression queries
- Useful for advanced code analysis but increases attack surface

When disabled, `ts.query` returns Problem Details (403) with instructions to enable.

## Examples

### Custom Repository Root

```bash
export KGF_REPO_ROOT=/path/to/my/repo
python -m codeintel.mcp_server.server
```

### Enable Advanced Queries

```bash
export CODEINTEL_ENABLE_TS_QUERY=1
python -m codeintel.cli mcp serve --repo .
```

### Build Persistent Index

```bash
# Incremental (default)
python -m codeintel.cli index build --repo .

# Fresh rebuild
python -m codeintel.cli index build --repo . --fresh
```

