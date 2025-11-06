# Command Reference

# app

KGFoundry Orchestration (0.1.0)

Usage:

```
 [OPTIONS] COMMAND [ARGS]...
```

## api

Launch the FastAPI search service using uvicorn.

## Parameters

port : int Port to bind the server to. Defaults to 8080.

## Raises

typer.Exit Raised when the server cannot be started (missing uvicorn entrypoint or missing dependency). Envelopes record the failure metadata for downstream tooling.

Usage:

```
 api [OPTIONS]
```

Options:

```
  --port INTEGER  Port to bind  \[default: 8080]
```

## e2e

Execute the Prefect-powered end-to-end orchestration pipeline.

## Raises

typer.Exit Raised with a non-zero exit code when the pipeline cannot be executed (for example, Prefect is not installed). The envelope captures the associated Problem Details payload.

Usage:

```
 e2e [OPTIONS]
```

## index-bm25

Build a BM25 index from chunk metadata and emit a CLI envelope.

## Parameters

chunks_parquet : Annotated[str, typer.Argument] Path to Parquet/JSONL file with chunks. backend : Annotated[str, typer.Option], optional Backend to use: 'lucene' or 'pure'. Defaults to 'lucene'. index_dir : Annotated[str, typer.Option], optional Output index directory. Defaults to './\_indices/bm25'.

## Raises

typer.Exit Raised with a non-zero exit code when index construction fails. The generated envelope captures the associated Problem Details payload.

Usage:

```
 index-bm25 [OPTIONS] CHUNKS_PARQUET
```

Options:

```
  CHUNKS_PARQUET    Path to Parquet/JSONL with chunks  \[required]
  --backend TEXT    lucene|pure  \[default: lucene]
  --index-dir TEXT  Output index directory  \[default: ./_indices/bm25]
```

## index-faiss

Build a FAISS index and emit a structured CLI envelope.

## Parameters

dense_vectors : Annotated[str, typer.Argument] Path to the dense vector payload (JSON skeleton format). index_path : Annotated[str, typer.Option], optional Destination path for the serialized FAISS index. Defaults to './\_indices/faiss/shard_000.idx'. factory : Annotated[str, typer.Option], optional FAISS factory string describing index topology. Defaults to 'Flat'. metric : Annotated[str, typer.Option], optional Similarity metric identifier (`"ip"` or `"l2"`). Defaults to 'ip'.

## Raises

typer.Exit Raised with a non-zero exit code when the command fails. The envelope captures the associated Problem Details payload for downstream tooling.

## Examples

> > > orchestration_cli = **import**("orchestration.cli").cli orchestration_cli.index_faiss( # doctest: +SKIP ... "vectors.json", ... "./\_indices/faiss/shard_000.idx", ... factory="Flat", ... metric="ip", ... )

Usage:

```
 index-faiss [OPTIONS] DENSE_VECTORS
```

Options:

```
  DENSE_VECTORS      Path to dense vectors JSON (skeleton)  \[required]
  --index-path TEXT  Output FAISS index path  \[default:
                     ./_indices/faiss/shard_000.idx]
  --factory TEXT     FAISS factory string  \[default: Flat]
  --metric TEXT      Similarity metric ('ip' or 'l2')  \[default: ip]
```

## index_bm25

Build a BM25 index from chunk metadata and emit a CLI envelope.

## Parameters

chunks_parquet : Annotated[str, typer.Argument] Path to Parquet/JSONL file with chunks. backend : Annotated[str, typer.Option], optional Backend to use: 'lucene' or 'pure'. Defaults to 'lucene'. index_dir : Annotated[str, typer.Option], optional Output index directory. Defaults to './\_indices/bm25'.

## Raises

typer.Exit Raised with a non-zero exit code when index construction fails. The generated envelope captures the associated Problem Details payload.

Usage:

```
 index_bm25 [OPTIONS] CHUNKS_PARQUET
```

Options:

```
  CHUNKS_PARQUET    Path to Parquet/JSONL with chunks  \[required]
  --backend TEXT    lucene|pure  \[default: lucene]
  --index-dir TEXT  Output index directory  \[default: ./_indices/bm25]
```

## index_faiss

Build a FAISS index and emit a structured CLI envelope.

## Parameters

dense_vectors : Annotated[str, typer.Argument] Path to the dense vector payload (JSON skeleton format). index_path : Annotated[str, typer.Option], optional Destination path for the serialized FAISS index. Defaults to './\_indices/faiss/shard_000.idx'. factory : Annotated[str, typer.Option], optional FAISS factory string describing index topology. Defaults to 'Flat'. metric : Annotated[str, typer.Option], optional Similarity metric identifier (`"ip"` or `"l2"`). Defaults to 'ip'.

## Raises

typer.Exit Raised with a non-zero exit code when the command fails. The envelope captures the associated Problem Details payload for downstream tooling.

## Examples

> > > orchestration_cli = **import**("orchestration.cli").cli orchestration_cli.index_faiss( # doctest: +SKIP ... "vectors.json", ... "./\_indices/faiss/shard_000.idx", ... factory="Flat", ... metric="ip", ... )

Usage:

```
 index_faiss [OPTIONS] DENSE_VECTORS
```

Options:

```
  DENSE_VECTORS      Path to dense vectors JSON (skeleton)  \[required]
  --index-path TEXT  Output FAISS index path  \[default:
                     ./_indices/faiss/shard_000.idx]
  --factory TEXT     FAISS factory string  \[default: Flat]
  --metric TEXT      Similarity metric ('ip' or 'l2')  \[default: ip]
```

:::
