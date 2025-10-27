# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:1:1) | [view](base.py#L1)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:14:1) | [view](base.py#L14-L22)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:25:1) | [view](base.py#L25-L38)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:1:1) | [view](bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Bm25doc → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:24:1) | [view](bm25.py#L24-L30)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:196:1) | [view](bm25.py#L196-L281)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Simple offline BM25 builder & searcher (Okapi BM25) → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:33:1) | [view](bm25.py#L33-L193)
  - **`embeddings_sparse.bm25.get_bm25`** — Get bm25 → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:284:1) | [view](bm25.py#L284-L298)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:1:1) | [view](splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Pyserini SPLADE impact index wrapper → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:168:1) | [view](splade.py#L168-L207)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Toy 'impact' index that approximates SPLADE with IDF/log weighting → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:65:1) | [view](splade.py#L65-L165)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Spladev3encoder → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:23:1) | [view](splade.py#L23-L62)
  - **`embeddings_sparse.splade.get_splade`** — Get splade → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:210:1) | [view](splade.py#L210-L222)
