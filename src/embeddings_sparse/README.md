# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](./base.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/base.py#L1)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](./base.py:14:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/base.py#L14-L21)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](./base.py:24:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/base.py#L24-L35)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](./bm25.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Bm25doc → [open](./bm25.py:24:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/bm25.py#L24-L30)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter → [open](./bm25.py:220:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/bm25.py#L220-L322)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Simple offline BM25 builder & searcher (Okapi BM25) → [open](./bm25.py:33:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/bm25.py#L33-L217)
  - **`embeddings_sparse.bm25.get_bm25`** — Get bm25 → [open](./bm25.py:325:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/bm25.py#L325-L348)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](./splade.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Pyserini SPLADE impact index wrapper → [open](./splade.py:195:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/splade.py#L195-L243)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Toy 'impact' index that approximates SPLADE with IDF/log weighting → [open](./splade.py:75:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/splade.py#L75-L192)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Spladev3encoder → [open](./splade.py:23:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/splade.py#L23-L72)
  - **`embeddings_sparse.splade.get_splade`** — Get splade → [open](./splade.py:246:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/0158375af4dd3ecccea4eb2e12a69dcba52db136/src/embeddings_sparse/splade.py#L246-L261)
