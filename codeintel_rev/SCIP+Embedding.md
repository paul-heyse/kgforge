Absolutely—here’s a lean, reproducible “first-run” that goes end-to-end:

* index your repo with **scip-python** (writes a local `index.scip` artifact)
* pull out symbol definitions + ranges
* do **cAST-style** (split-then-merge) chunking from those ranges
* call **vLLM** serving **Nomic’s code embedding model** to embed each chunk
* save vectors to **Parquet** and also into **DuckDB** (no FAISS in this pass)

Yes: `scip-python index` writes a persistent `index.scip` file; you don’t need to mirror that whole thing into DuckDB—store only what you want (e.g., chunk UUIDs + vectors). ([Sourcegraph][1])

---

# 0) One-time setup

## Install the Python indexer (scip-python)

```bash
# Node 16+ required
npm install -g @sourcegraph/scip-python
# in your repo root, with deps installed for best results
scip-python index . --project-name "$REPO"
# you now have: ./index.scip
```

Sourcegraph’s quickstart shows this exact flow. ([Sourcegraph][1])

## (Optional) Install the `scip` CLI to JSON-dump the index

```bash
# build the CLI (Go) from the scip repo
git clone https://github.com/sourcegraph/scip.git
cd scip && go build ./cmd/scip
# print the protobuf index as JSON
./scip print --json ../index.scip > ../index.scip.json
```

The repo ships the CLI and the Protobuf schema; `scip print --json` is the intended local-consumption path. ([GitHub][2])

> Prefer JSON for a first run—parsing the `.proto` directly is easy later if you want (the schema is in the repo). ([GitHub][2])

## Serve the embedding model with **vLLM**

Run the **OpenAI-compatible** server in embedding mode with **Nomic’s code embedder**:

```bash
pip install -U vllm
python -m vllm.entrypoints.openai.api_server \
  --model nomic-ai/nomic-embed-code \
  --task embed \
  --dtype bfloat16 \
  --trust-remote-code \
  --port 8000
```

vLLM supports pooling/embedding models (`--task embed`) and provides an OpenAI-compatible `/v1/embeddings` API. If a custom model requires special pooling, `--trust-remote-code` helps. ([VLLM Docs][3])
The **Nomic Embed Code** model card (7B, code-focused) is here. ([Hugging Face][4])

---

# 1) Python: read `index.scip.json` → make cAST-style chunks → embed → Parquet + DuckDB

Drop this file as `first_run_scip_embed.py` in the repo root and run it after the steps above.

```python
# first_run_scip_embed.py
import json, os, uuid, time, hashlib, requests, duckdb, pathlib
from typing import List, Dict, Any, Tuple
from collections import defaultdict

REPO_ROOT = pathlib.Path(".").resolve()
SCIP_JSON = REPO_ROOT / "index.scip.json"
EMBED_URL = "http://localhost:8000/v1/embeddings"   # vLLM OpenAI-compatible
EMBED_MODEL = "nomic-ai/nomic-embed-code"           # served above
PARQUET_OUT = REPO_ROOT / "code_embeddings.parquet"
DUCK = REPO_ROOT / "code.duckdb"

# ---------- small helpers ----------
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def v5(ns: uuid.UUID, s: str) -> str:
    return uuid.uuid5(ns, s).hex

def line_starts(text: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts

def to_off(starts: List[int], line: int, col: int, N: int) -> int:
    if line >= len(starts): return N
    return min(starts[line] + col, N)

def load_scip(path: pathlib.Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ---------- parse SCIP JSON for definition occurrences ----------
def extract_definitions(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return list of definition occurrences with:
      {symbol, start_line, start_char, end_line, end_char}
    We treat 'Definition' role bit as (roles & 1) != 0; if roles missing,
    fall back to first occurrence per symbol.
    """
    occs = doc.get("occurrences", [])
    defs = []
    first_by_symbol = {}
    for o in occs:
        rng = o.get("range") or o.get("enclosingRange") or []
        # SCIP JSON typically stores range as [sl, sc, el, ec]
        if isinstance(rng, list) and len(rng) == 4:
            sl, sc, el, ec = rng
        elif isinstance(rng, dict):
            s = rng.get("start", {}); e = rng.get("end", {})
            sl, sc, el, ec = s.get("line", 0), s.get("character", 0), e.get("line", 0), e.get("character", 0)
        else:
            continue
        symbol = o.get("symbol")
        roles = o.get("symbolRoles") or o.get("symbol_roles") or 0
        is_def = False
        if isinstance(roles, int):
            is_def = (roles & 1) != 0      # bit 0 => Definition in common indexers
        elif isinstance(roles, list):
            # sometimes roles can be names like "DEFINITION" or numbers
            is_def = any((r == 1) or (r == "DEFINITION") for r in roles)
        if is_def:
            defs.append({"symbol": symbol, "sl": sl, "sc": sc, "el": el, "ec": ec})
        if symbol and symbol not in first_by_symbol:
            first_by_symbol[symbol] = {"symbol": symbol, "sl": sl, "sc": sc, "el": el, "ec": ec}
    if defs:
        return defs
    # fallback: first occurrence per symbol as "definition"
    return list(first_by_symbol.values())

def not_nested(intervals: List[Tuple[int,int]]) -> List[int]:
    """Return indices of intervals that are not contained by others (top-level in a file)."""
    out = []
    for i,(s1,e1) in enumerate(intervals):
        if all(not (s2 <= s1 and e1 <= e2) or i==j for j,(s2,e2) in enumerate(intervals)):
            out.append(i)
    return out

# ---------- cAST-ish: greedy pack of top-level defs; split large nodes on blank lines ----------
def chunk_file(document: Dict[str,Any], repo_root: pathlib.Path, budget_chars: int = 2000) -> List[Dict[str, Any]]:
    rel = document.get("relativePath") or document.get("relative_path") or ""
    path = repo_root / rel
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    N = len(text); starts = line_starts(text)

    defs = extract_definitions(document)
    # map SCIP def ranges to byte offsets
    spans = []
    for d in defs:
        s = to_off(starts, d["sl"], d["sc"], N)
        e = to_off(starts, d["el"], d["ec"], N)
        spans.append((d["symbol"], s, e))

    # choose "top-level" defs (not contained by other defs), then sort by start
    intervals = [(s,e) for _,s,e in spans]
    top_idx = not_nested(intervals)
    tops = [spans[i] for i in top_idx]
    tops.sort(key=lambda t: t[1])

    chunks = []
    cur_s, cur_e, cur_syms = None, None, []
    for sym, s, e in tops:
        piece_len = e - s
        if cur_s is None:
            cur_s, cur_e, cur_syms = s, e, [sym]
            continue
        if (max(cur_e, e) - cur_s) <= budget_chars:
            cur_e = max(cur_e, e); cur_syms.append(sym)
        else:
            # flush current
            chunks.append((cur_s, cur_e, cur_syms))
            # if single symbol is huge, split on blank lines ~ budget
            if piece_len > budget_chars:
                start = s
                while start < e:
                    end = min(start + budget_chars, e)
                    # advance to nearest blank line boundary if possible
                    nl = text.rfind("\n\n", start, end)
                    if nl != -1 and nl > start + budget_chars//2:
                        end = nl + 1
                    chunks.append((start, end, [sym]))
                    start = end
                cur_s = cur_e = None; cur_syms = []
            else:
                cur_s, cur_e, cur_syms = s, e, [sym]
    if cur_s is not None:
        chunks.append((cur_s, cur_e, cur_syms))

    # finalize payloads
    out = []
    for s,e,syms in chunks:
        merged = text[s:e]
        # stable chunk UUID from file + symbol set
        chunk_uuid = v5(uuid.NAMESPACE_URL, f"{str(path)}|{','.join(sorted(syms))}|v1")
        out.append({
            "file": str(path),
            "start": s, "end": e,
            "symbols": syms,
            "chunk_uuid": chunk_uuid,
            "text": merged,
            "text_sha256": sha256(merged),
        })
    return out

# ---------- embed via vLLM OpenAI-compatible /v1/embeddings ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    data = r.json()["data"]
    # OpenAI-compatible: data[i].embedding
    return [row["embedding"] for row in data]

# ---------- run ----------
def main():
    scip = load_scip(SCIP_JSON)
    docs = scip.get("documents", [])
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_file(doc, REPO_ROOT, budget_chars=2000))

    # batch embeds
    B = 32
    vectors = {}
    for i in range(0, len(all_chunks), B):
        batch = all_chunks[i:i+B]
        vecs = embed_texts([c["text"] for c in batch])
        for c, v in zip(batch, vecs):
            vectors[c["chunk_uuid"]] = v

    # write Parquet (via DuckDB, simplest)
    con = duckdb.connect(DUCK)
    con.execute("CREATE OR REPLACE TABLE _tmp_chunks AS SELECT * FROM (VALUES (1)) WHERE 1=0")
    # build an in-memory table from Python lists
    con.execute("""
        CREATE OR REPLACE TABLE embeddings AS
        SELECT
          chunk_uuid,
          file,
          start,
          "end",
          symbols::TEXT,      -- store symbols as TEXT (CSV) for now
          text_sha256,
          ?::VARCHAR as model,
          ?::TIMESTAMP as created_at,
          ?::FLOAT[] as embedding
        FROM (SELECT 1) WHERE 1=0
    """, [EMBED_MODEL, time.strftime("%Y-%m-%d %H:%M:%S"), [0.0]])
    # insert rows
    for c in all_chunks:
        emb = vectors[c["chunk_uuid"]]
        con.execute("""
          INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, now(), ?)
        """, [
          c["chunk_uuid"], c["file"], c["start"], c["end"],
          ",".join(c["symbols"]), c["text_sha256"], EMBED_MODEL, emb
        ])
    # also export as Parquet
    con.execute(f"COPY (SELECT * FROM embeddings) TO '{PARQUET_OUT.as_posix()}' (FORMAT PARQUET)")
    print(f"Wrote {len(all_chunks)} chunks → {PARQUET_OUT}")

if __name__ == "__main__":
    main()
```

**What this does**

* Treats **SCIP symbol strings** as your **permanent IDs**; and, if you want UUIDs, it creates deterministic **UUIDv5** from the symbol set for each chunk (the SCIP string is the durable “name”). SCIP’s symbol strings are the canonical, stable identifiers by design. ([GitHub][2])
* Implements a **cAST-style** policy: keep top-level definitions intact, greedily pack siblings up to a budget, and split over-long items on natural boundaries. This mirrors the “recursive split, greedy merge” idea from the CAST paper (we’ve kept it simple for a first run). ([arXiv][5])
* Calls **vLLM**’s OpenAI-compatible **/v1/embeddings** endpoint (served with `--task embed`) so you can swap models later without changing the client. ([VLLM Docs][6])
* Writes your embeddings to **Parquet** and imports them into **DuckDB** (you can also query Parquet directly if you prefer). ([DuckDB][7])

> If you’d rather decode `index.scip` directly (no JSON step), generate Python classes from `scip.proto` and parse—SCIP ships the schema and CLI together. ([GitHub][2])

---

## 2) Inspecting and using what you produced

* The persistent artifact is **`index.scip`**. That’s the one file scip-python creates; it’s your durable source of truth for symbol strings + occurrences. Keep it in build artifacts or alongside the commit you indexed. ([Sourcegraph][1])
* Your vectors live in `code_embeddings.parquet` and in DuckDB `code.duckdb` table `embeddings`. DuckDB’s Parquet support lets you either **query the file directly** or **CTAS** to make a physical table. ([DuckDB][7])

---

## 3) Notes & gotchas

* **vLLM + Nomic**: vLLM’s **pooling models** + OpenAI-compatible embeddings are documented. If this particular checkpoint ever errors (pooling adapter mismatch), two proven workarounds are (a) run **SentenceTransformers** client-side for the very first batch, or (b) use another supported embedder (e.g., E5) to validate the pipeline—then swap back. ([VLLM Docs][3])
* **Symbol roles**: the code assumes the **Definition** bit is set as `roles & 1`. That’s the common mapping; if you see weirdness in your data, we can flip the filter to “first occurrence per symbol” for the first run and refine from there. (SCIP’s CLI/issue threads discuss using `scip print --json` for local tools.) ([GitHub][8])
* **“Do we need DuckDB for the index?”** No. Keep **only vectors + lightweight metadata** in DuckDB. The **SCIP file is your persisted graph**. If later you want a full graph DB with extra fields, you can ingest the same `index.scip` into **Glean**; it accepts SCIP. ([Glean][9])

---

### References

* scip-python quickstart (install & `index` producing `index.scip`). ([Sourcegraph][1])
* SCIP repo (schema + CLI). ([GitHub][2])
* **CAST** structural chunking (recursive split + merge siblings). ([arXiv][5])
* vLLM embeddings & OpenAI-compatible server. ([VLLM Docs][10])
* Nomic **code** embedding model. ([Hugging Face][4])
* DuckDB Parquet read/CTAS. ([DuckDB][7])

If you want, I can tweak the chunker to recurse into **class → method** boundaries (closer to the paper), or add a quick **`CREATE TABLE ... FLOAT[dim]`** if you plan to use DuckDB’s VSS later.

[1]: https://sourcegraph.com/blog/scip-python?utm_source=chatgpt.com "scip-python: a precise Python indexer"
[2]: https://github.com/sourcegraph/scip "GitHub - sourcegraph/scip: SCIP Code Intelligence Protocol"
[3]: https://docs.vllm.ai/en/latest/models/pooling_models.html?utm_source=chatgpt.com "Pooling Models - vLLM"
[4]: https://huggingface.co/nomic-ai/nomic-embed-code "nomic-ai/nomic-embed-code · Hugging Face"
[5]: https://arxiv.org/abs/2506.15655?utm_source=chatgpt.com "cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree"
[6]: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
[7]: https://duckdb.org/docs/stable/data/parquet/overview.html?utm_source=chatgpt.com "Reading and Writing Parquet Files"
[8]: https://github.com/sourcegraph/scip/issues/178?utm_source=chatgpt.com "Using index.scip locally? · Issue #178 · sourcegraph/scip"
[9]: https://glean.software/docs/indexer/scip-python/?utm_source=chatgpt.com "Python - Glean"
[10]: https://docs.vllm.ai/en/v0.7.0/getting_started/examples/embedding.html?utm_source=chatgpt.com "Embedding — vLLM"
