# CodeIntel MCP - Best-in-Class Code Intelligence Platform

Production-grade MCP server for AI-assisted code review with SCIP indexing, semantic search, and HTTP/3 streaming.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Client (ChatGPT, Claude, etc via MCP)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTPS (H2/H3)
┌──────────────────────▼──────────────────────────────────────┐
│  NGINX-QUIC (Edge)                                          │
│  - HTTP/3 + H2                                              │
│  - OAuth 2.1 (optional)                                     │
│  - Streaming (proxy_buffering off)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Hypercorn (ASGI)                                           │
│  - HTTP/2, HTTP/3                                           │
│  - Backpressure-aware streaming                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI + FastMCP                                          │
│  - /healthz, /readyz, /sse                                  │
│  - /mcp/* (Streamable HTTP)                                 │
│  - Tool catalog (search, symbols, history, etc)             │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │   FAISS     │     │   DuckDB    │     │    vLLM     │
    │   (GPU)     │     │  (Parquet)  │     │ (Embeddings)│
    └─────────────┘     └─────────────┘     └─────────────┘
```

## Features

- **SCIP-based Indexing**: Uses Sourcegraph SCIP for precise symbol information
- **cAST Chunking**: Structure-aware code chunking (2200 char budget)
- **Semantic Search**: GPU FAISS with cuVS acceleration
- **Hybrid Retrieval**: RRF fusion (future: BM25 + SPLADE + embeddings)
- **HTTP/3 Streaming**: QUIC + proper backpressure for token streams
- **FastMCP Tools**: Full QueryScope catalog (search, symbols, history, docs)
- **Production Edge**: NGINX with OAuth 2.1, streaming optimizations

## Prerequisites

### System Requirements

- Python 3.13.9
- CUDA-capable GPU (for FAISS + vLLM)
- Node.js 16+ (for SCIP Python indexer)
- NGINX with QUIC support (or nginx-quic build)
- 16GB+ RAM recommended
- 10GB+ disk for indexes

### Dependencies

```bash
# Core dependencies (already in environment via bootstrap.sh)
uv add msgspec httpx numpy pyarrow duckdb fastapi starlette fastmcp

# GPU stack (optional dependencies)
uv add --group gpu faiss vllm torch libcuvs-cu13

# SCIP indexer
npm install -g @sourcegraph/scip-python
```

## Quick Start

### 1. Bootstrap Environment

```bash
cd /home/paul/kgfoundry
scripts/bootstrap.sh
```

This sets up Python 3.13.9, syncs dependencies, and activates the `.venv`.

### 2. Generate SCIP Index

```bash
cd codeintel-rev
scip-python index ../src --project-name kgfoundry
# Export to JSON for parsing
# (Optional: install scip CLI from github.com/sourcegraph/scip)
# scip print --json index.scip > index.scip.json
```

### 3. Start vLLM Embedding Service

```bash
# Start vLLM with Nomic code embeddings
vllm serve nomic-ai/nomic-embed-code \
  --task embed \
  --dtype bfloat16 \
  --trust-remote-code \
  --port 8001
```

### 4. Run Indexing Pipeline

```bash
cd codeintel-rev

# Set environment
export REPO_ROOT=/home/paul/kgfoundry
export SCIP_INDEX=index.scip.json
export VLLM_URL=http://127.0.0.1:8001/v1

# Run indexer
python bin/index_all.py
```

This will:
1. Parse SCIP index
2. Chunk files using cAST (symbol boundaries)
3. Embed chunks with vLLM
4. Write Parquet with vectors
5. Build FAISS GPU index
6. Initialize DuckDB catalog

### 5. Start MCP Server

```bash
cd codeintel-rev

# Option A: Direct with Hypercorn
hypercorn --config app/hypercorn.toml codeintel_rev.app.main:app

# Option B: Development mode
uvicorn codeintel_rev.app.main:app --reload --port 8000
```

### 6. Configure NGINX (Production)

```bash
# Copy NGINX config
sudo cp config/nginx/codeintel-mcp.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/codeintel-mcp.conf /etc/nginx/sites-enabled/

# Update server_name and paths in config
sudo vim /etc/nginx/sites-available/codeintel-mcp.conf

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

### 7. Test MCP Server

```bash
# Health check
curl https://localhost:8000/healthz

# Readiness check
curl https://localhost:8000/readyz

# Test SSE streaming
curl https://localhost:8000/sse

# Test MCP tools (requires MCP client)
# Use FastMCP dev server:
mcp dev codeintel_rev/mcp_server/server.py
```

## Configuration

All configuration is via environment variables:

### Core Settings

```bash
export REPO_ROOT=/path/to/repo
export DATA_DIR=data
export VECTORS_DIR=data/vectors
export FAISS_INDEX=data/faiss/code.ivfpq.faiss
export DUCKDB_PATH=data/catalog.duckdb
export SCIP_INDEX=index.scip.json
```

### vLLM Settings

```bash
export VLLM_URL=http://127.0.0.1:8001/v1
export VLLM_MODEL=nomic-ai/nomic-embed-code
export VLLM_BATCH_SIZE=64
export VLLM_TIMEOUT_S=120.0
```

### Index Settings

```bash
export VEC_DIM=2560              # Nomic embed dimension
export CHUNK_BUDGET=2200         # cAST chunk size
export FAISS_NLIST=8192          # IVF centroids
export FAISS_NPROBE=128          # Search probes
export USE_CUVS=1                # Enable cuVS acceleration
```

### Server Limits

```bash
export MAX_RESULTS=1000
export QUERY_TIMEOUT_S=30.0
export RATE_LIMIT_QPS=10.0
export RATE_LIMIT_BURST=20
```

## MCP Tools

The server exposes the following tools via FastMCP:

### Scope & Navigation
- `set_scope(scope)` - Set query scope
- `list_paths(path, globs, max)` - List files
- `open_file(path, start_line, end_line)` - Read file content

### Search
- `search_text(query, regex, paths)` - Fast text search
- `semantic_search(query, limit)` - Semantic code search (FAISS)

### Symbols
- `symbol_search(query, kind, language)` - Find symbols
- `definition_at(path, line, char)` - Go to definition
- `references_at(path, line, char)` - Find references

### Git History
- `blame_range(path, start_line, end_line)` - Git blame
- `file_history(path, limit)` - Commit history

### Resources
- `file://{path}` - File content as MCP resource

### Prompts
- `prompt_code_review(area)` - Code review template

## Development

### Project Structure

```
codeintel-rev/
├── app/                  # FastAPI application
├── bin/                  # CLI scripts (index_all.py)
├── config/               # Configuration files
│   └── nginx/            # NGINX configs
├── indexing/             # SCIP reader, cAST chunker
├── io/                   # Storage (Parquet, DuckDB, FAISS, vLLM)
├── mcp_server/           # FastMCP server + tools
├── retrieval/            # Hybrid search (RRF)
└── tests/                # Tests
```

### Running Tests

```bash
# Unit tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=codeintel_rev --cov-report=html
```

### Quality Gates

```bash
# Format and lint
uv run ruff format codeintel-rev/
uv run ruff check --fix codeintel-rev/

# Type check
uv run pyright codeintel-rev/
uv run pyrefly check

# Security scan
uv run pip-audit
```

## OAuth 2.1 Setup (Optional)

### Using oauth2-proxy

```bash
# Install oauth2-proxy
wget https://github.com/oauth2-proxy/oauth2-proxy/releases/download/v7.4.0/oauth2-proxy-v7.4.0.linux-amd64.tar.gz
tar -xzf oauth2-proxy-v7.4.0.linux-amd64.tar.gz

# Configure
cat > oauth2-proxy.cfg <<EOF
http_address = "127.0.0.1:4180"
upstreams = ["http://127.0.0.1:8000"]
provider = "oidc"
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
oidc_issuer_url = "https://YOUR_IDP/.well-known/openid-configuration"
cookie_secret = "$(openssl rand -base64 32)"
email_domains = ["yourdomain.com"]
EOF

# Run
./oauth2-proxy --config oauth2-proxy.cfg
```

### NGINX OAuth Integration

Uncomment the `auth_request` directives in `config/nginx/codeintel-mcp.conf`.

## Performance Tuning

### FAISS

```bash
# Adjust IVF parameters
export FAISS_NLIST=16384  # More centroids = better recall, slower training
export FAISS_NPROBE=256   # More probes = better recall, slower search

# Enable cuVS (requires custom FAISS wheel)
export USE_CUVS=1
```

### vLLM

```bash
# Increase batch size for throughput
export VLLM_BATCH_SIZE=128

# Enable tensor parallelism for large models
vllm serve ... --tensor-parallel-size 2
```

### DuckDB

```bash
# Increase memory limit
export DUCKDB_MEMORY_LIMIT=8GB

# Enable parallel query execution
export DUCKDB_THREADS=8
```

## Troubleshooting

### FAISS GPU fails

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Fallback to CPU-only FAISS
export USE_CUVS=0
# Or install CPU-only FAISS
uv add faiss-cpu
```

### vLLM connection issues

```bash
# Test vLLM directly
curl http://localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"nomic-ai/nomic-embed-code","input":"test"}'
```

### HTTP/3 not working

```bash
# Verify QUIC/UDP port open
sudo ufw allow 443/udp

# Check NGINX QUIC module
nginx -V 2>&1 | grep quic

# Test with curl
curl --http3 -I https://localhost/healthz
```

## License

See main project LICENSE.

## References

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [SCIP Protocol](https://github.com/sourcegraph/scip)
- [Nomic Embed Code](https://huggingface.co/nomic-ai/nomic-embed-code)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [AGENTS.md](../AGENTS.md) - Development standards

