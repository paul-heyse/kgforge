# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](./base.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/base.py#L1)
  - **`embeddings_sparse.base.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](./base.py:33:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/base.py#L33-L40)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](./base.py:44:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/base.py#L44-L55)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](./bm25.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Serialized representation of a BM25 indexed document → [open](./bm25.py:72:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/bm25.py#L72-L78)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter lazily importing Lucene bindings → [open](./bm25.py:200:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/bm25.py#L200-L261)
  - **`embeddings_sparse.bm25.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Pure-Python BM25 builder and searcher for sparse retrieval → [open](./bm25.py:82:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/bm25.py#L82-L196)
  - **`embeddings_sparse.bm25.get_bm25`** — Construct a BM25 implementation for the requested backend → [open](./bm25.py:265:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/bm25.py#L265-L280)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](./splade.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Bridge to a Pyserini SPLADE impact index stored on disk → [open](./splade.py:176:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/splade.py#L176-L202)
  - **`embeddings_sparse.splade.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Approximate SPLADE indexing with TF/IDF-style impact weighting → [open](./splade.py:103:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/splade.py#L103-L172)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Describe the SPLADE configuration used for neural encoding → [open](./splade.py:47:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/splade.py#L47-L99)
  - **`embeddings_sparse.splade.get_splade`** — Construct a SPLADE impact index for the requested backend → [open](./splade.py:206:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/88ccab0c57ccecf30fc5b8829a70ebdc05634b35/src/embeddings_sparse/splade.py#L206-L213)
