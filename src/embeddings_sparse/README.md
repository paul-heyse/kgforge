# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/base.py?line=1&column=1) | [view](./base.py#L1)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/base.py?line=14&column=1) | [view](./base.py#L14-L21)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/base.py?line=24&column=1) | [view](./base.py#L24-L35)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/bm25.py?line=1&column=1) | [view](./bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Bm25doc → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/bm25.py?line=24&column=1) | [view](./bm25.py#L24-L30)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/bm25.py?line=220&column=1) | [view](./bm25.py#L220-L322)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Simple offline BM25 builder & searcher (Okapi BM25) → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/bm25.py?line=33&column=1) | [view](./bm25.py#L33-L217)
  - **`embeddings_sparse.bm25.get_bm25`** — Get bm25 → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/bm25.py?line=325&column=1) | [view](./bm25.py#L325-L348)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/splade.py?line=1&column=1) | [view](./splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Pyserini SPLADE impact index wrapper → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/splade.py?line=195&column=1) | [view](./splade.py#L195-L243)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Toy 'impact' index that approximates SPLADE with IDF/log weighting → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/splade.py?line=75&column=1) | [view](./splade.py#L75-L192)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Spladev3encoder → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/splade.py?line=23&column=1) | [view](./splade.py#L23-L72)
  - **`embeddings_sparse.splade.get_splade`** — Get splade → [open](vscode://file/home/paul/kgfoundry/src/embeddings_sparse/splade.py?line=246&column=1) | [view](./splade.py#L246-L261)
