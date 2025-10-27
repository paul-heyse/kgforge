# `embeddings_sparse`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`embeddings_sparse.base`** — Module for embeddings_sparse.base → [open](./base.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/base.py#L1)
  - **`embeddings_sparse.base.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.base.SparseEncoder`** — Protocol for sparse text encoders → [open](./base.py:32:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/base.py#L32-L39)
  - **`embeddings_sparse.base.SparseIndex`** — Protocol describing sparse index interactions → [open](./base.py:43:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/base.py#L43-L54)
- **`embeddings_sparse.bm25`** — Module for embeddings_sparse.bm25 → [open](./bm25.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/bm25.py#L1)
  - **`embeddings_sparse.bm25.BM25Doc`** — Serialized representation of a BM25 indexed document → [open](./bm25.py:73:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/bm25.py#L73-L79)
  - **`embeddings_sparse.bm25.LuceneBM25`** — Pyserini-backed Lucene BM25 adapter lazily importing Lucene bindings → [open](./bm25.py:199:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/bm25.py#L199-L260)
  - **`embeddings_sparse.bm25.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.bm25.PurePythonBM25`** — Pure-Python BM25 builder and searcher for sparse retrieval → [open](./bm25.py:82:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/bm25.py#L82-L196)
  - **`embeddings_sparse.bm25.get_bm25`** — Return a backend-specific BM25 implementation → [open](./bm25.py:263:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/bm25.py#L263-L278)
- **`embeddings_sparse.splade`** — Module for embeddings_sparse.splade → [open](./splade.py:1:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/splade.py#L1)
  - **`embeddings_sparse.splade.LuceneImpactIndex`** — Pyserini SPLADE impact index wrapper → [open](./splade.py:221:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/splade.py#L221-L269)
  - **`embeddings_sparse.splade.NavMap`** — Structure describing a module navmap → [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/kgfoundry_common/navmap_types.py#L38-L51)
  - **`embeddings_sparse.splade.PureImpactIndex`** — Toy 'impact' index that approximates SPLADE with IDF/log weighting → [open](./splade.py:100:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/splade.py#L100-L217)
  - **`embeddings_sparse.splade.SPLADEv3Encoder`** — Spladev3encoder → [open](./splade.py:47:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/splade.py#L47-L96)
  - **`embeddings_sparse.splade.get_splade`** — Get splade → [open](./splade.py:273:1) | [view](https://github.com/paul-heyse/kgfoundry/blob/98bc876ceda60da4a9d9e9a3642946ff0b0447e3/src/embeddings_sparse/splade.py#L273-L288)
