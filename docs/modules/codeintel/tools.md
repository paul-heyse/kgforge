# CodeIntel MCP Tools

Auto-generated from server schemas. Do not edit manually.


## `ts.query`

Run a Tree-sitter query against a file.

**Parameters**:

- `path`: `string` *(required)* — Absolute or repo-relative file path
- `language`: `string` — Tree-sitter language identifier
- `query`: `string` *(required)* — Tree-sitter S-expression to execute


## `ts.symbols`

List Python symbol definitions in a directory.

**Parameters**:

- `directory`: `string` *(required)* — Directory to scan for Python modules


## `ts.calls`

Enumerate call expressions within a directory.

**Parameters**:

- `directory`: `string` *(required)* — Directory to scan for call edges
- `language`: `string` — Language to analyse
- `callee`: `any` — Optional callee name filter


## `ts.errors`

Report syntax errors detected by Tree-sitter.

**Parameters**:

- `path`: `string` *(required)* — File to analyse for syntax errors
- `language`: `string` — Language to analyse


## `code.listFiles`

List repo files with optional filters.

**Parameters**:

- `directory`: `any` — Directory to scan, or None for root
- `glob`: `any` — Optional glob pattern filter
- `limit`: `any` — Maximum number of files to return


## `code.getFile`

Read a file segment (UTF-8).

**Parameters**:

- `path`: `string` *(required)* — Repository-relative file path
- `offset`: `integer` — Byte offset to start reading
- `length`: `any` — Maximum bytes to read


## `code.getOutline`

Return an outline (functions/classes) for a file.

**Parameters**:

- `path`: `string` *(required)* — Repository-relative file path
- `language`: `string` — Tree-sitter language identifier


## `code.getAST`

Return a bounded AST snapshot.

**Parameters**:

- `path`: `string` *(required)* — Repository-relative file path
- `language`: `string` — Tree-sitter language identifier
- `format`: `string` — Output format: json or sexpr


## `code.health`

Return server health status.

**Parameters**: None

