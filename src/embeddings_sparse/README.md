# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [API](#api)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:1:1) | [view](base.py#L1)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:9:1) | [view](base.py#L9-L17)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/base.py:20:1) | [view](base.py#L20-L33)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:1:1) | [view](bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Bm25doc → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:17:1) | [view](bm25.py#L17-L23)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:189:1) | [view](bm25.py#L189-L274)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Simple offline BM25 builder & searcher (Okapi BM25) → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:26:1) | [view](bm25.py#L26-L186)
  - **`embeddings_sparse.bm25.get_bm25`** — Get bm25 → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/bm25.py:277:1) | [view](bm25.py#L277-L291)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:1:1) | [view](splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Pyserini SPLADE impact index wrapper → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:161:1) | [view](splade.py#L161-L200)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Toy 'impact' index that approximates SPLADE with IDF/log weighting → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:58:1) | [view](splade.py#L58-L158)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Spladev3encoder → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:16:1) | [view](splade.py#L16-L55)
  - **`embeddings_sparse.splade.get_splade`** — Get splade → [open](vscode://file//home/paul/KGForge/src/embeddings_sparse/splade.py:203:1) | [view](splade.py#L203-L215)
