# `search_api`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [API](#api)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`search_api.app`** — Module for search_api.app → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:1:1) | [view](app.py#L1)
  - **`search_api.app.SearchRequest`** — Searchrequest → [open](vscode://file//home/paul/KGForge/src/search_api/schemas.py:8:1) | [view](schemas.py#L8-L14)
  - **`search_api.app.SearchResult`** — Searchresult → [open](vscode://file//home/paul/KGForge/src/search_api/schemas.py:17:1) | [view](schemas.py#L17-L27)
  - **`search_api.app.apply_kg_boosts`** — Apply kg boosts → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:126:1) | [view](app.py#L126-L160)
  - **`search_api.app.auth`** — Auth → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:76:1) | [view](app.py#L76-L91)
  - **`search_api.app.graph_concepts`** — Graph concepts → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:223:1) | [view](app.py#L223-L238)
  - **`search_api.app.healthz`** — Healthz → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:94:1) | [view](app.py#L94-L106)
  - **`search_api.app.rrf_fuse`** — Rrf fuse → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:109:1) | [view](app.py#L109-L123)
  - **`search_api.app.search`** — Search → [open](vscode://file//home/paul/KGForge/src/search_api/app.py:163:1) | [view](app.py#L163-L220)
- **`search_api.bm25_index`** — Module for search_api.bm25_index → [open](vscode://file//home/paul/KGForge/src/search_api/bm25_index.py:1:1) | [view](bm25_index.py#L1)
  - **`search_api.bm25_index.BM25Doc`** — Bm25doc → [open](vscode://file//home/paul/KGForge/src/search_api/bm25_index.py:29:1) | [view](bm25_index.py#L29-L38)
  - **`search_api.bm25_index.BM25Index`** — Bm25index → [open](vscode://file//home/paul/KGForge/src/search_api/bm25_index.py:41:1) | [view](bm25_index.py#L41-L211)
  - **`search_api.bm25_index.toks`** — Toks → [open](vscode://file//home/paul/KGForge/src/search_api/bm25_index.py:17:1) | [view](bm25_index.py#L17-L26)
- **`search_api.faiss_adapter`** — Module for search_api.faiss_adapter → [open](vscode://file//home/paul/KGForge/src/search_api/faiss_adapter.py:1:1) | [view](faiss_adapter.py#L1)
  - **`search_api.faiss_adapter.DenseVecs`** — Densevecs → [open](vscode://file//home/paul/KGForge/src/search_api/faiss_adapter.py:27:1) | [view](faiss_adapter.py#L27-L32)
  - **`search_api.faiss_adapter.FaissAdapter`** — Faissadapter → [open](vscode://file//home/paul/KGForge/src/search_api/faiss_adapter.py:35:1) | [view](faiss_adapter.py#L35-L172)
- **`search_api.fixture_index`** — Module for search_api.fixture_index → [open](vscode://file//home/paul/KGForge/src/search_api/fixture_index.py:1:1) | [view](fixture_index.py#L1)
  - **`search_api.fixture_index.FixtureDoc`** — Fixturedoc → [open](vscode://file//home/paul/KGForge/src/search_api/fixture_index.py:27:1) | [view](fixture_index.py#L27-L35)
  - **`search_api.fixture_index.FixtureIndex`** — Fixtureindex → [open](vscode://file//home/paul/KGForge/src/search_api/fixture_index.py:38:1) | [view](fixture_index.py#L38-L154)
  - **`search_api.fixture_index.tokenize`** — Tokenize → [open](vscode://file//home/paul/KGForge/src/search_api/fixture_index.py:15:1) | [view](fixture_index.py#L15-L24)
- **`search_api.fusion`** — Module for search_api.fusion → [open](vscode://file//home/paul/KGForge/src/search_api/fusion.py:1:1) | [view](fusion.py#L1)
  - **`search_api.fusion.rrf_fuse`** — Rrf fuse → [open](vscode://file//home/paul/KGForge/src/search_api/fusion.py:6:1) | [view](fusion.py#L6-L20)
- **`search_api.kg_mock`** — Module for search_api.kg_mock → [open](vscode://file//home/paul/KGForge/src/search_api/kg_mock.py:1:1) | [view](kg_mock.py#L1)
  - **`search_api.kg_mock.detect_query_concepts`** — Detect query concepts → [open](vscode://file//home/paul/KGForge/src/search_api/kg_mock.py:17:1) | [view](kg_mock.py#L17-L31)
  - **`search_api.kg_mock.kg_boost`** — Kg boost → [open](vscode://file//home/paul/KGForge/src/search_api/kg_mock.py:51:1) | [view](kg_mock.py#L51-L68)
  - **`search_api.kg_mock.linked_concepts_for_text`** — Linked concepts for text → [open](vscode://file//home/paul/KGForge/src/search_api/kg_mock.py:34:1) | [view](kg_mock.py#L34-L48)
- **`search_api.schemas`** — Module for search_api.schemas → [open](vscode://file//home/paul/KGForge/src/search_api/schemas.py:1:1) | [view](schemas.py#L1)
  - **`search_api.schemas.SearchRequest`** — Searchrequest → [open](vscode://file//home/paul/KGForge/src/search_api/schemas.py:8:1) | [view](schemas.py#L8-L14)
  - **`search_api.schemas.SearchResult`** — Searchresult → [open](vscode://file//home/paul/KGForge/src/search_api/schemas.py:17:1) | [view](schemas.py#L17-L27)
- **`search_api.service`** — Module for search_api.service → [open](vscode://file//home/paul/KGForge/src/search_api/service.py:1:1) | [view](service.py#L1)
  - **`search_api.service.apply_kg_boosts`** — Apply kg boosts → [open](vscode://file//home/paul/KGForge/src/search_api/service.py:14:1) | [view](service.py#L14-L25)
  - **`search_api.service.mmr_deduplicate`** — Mmr deduplicate → [open](vscode://file//home/paul/KGForge/src/search_api/service.py:28:1) | [view](service.py#L28-L41)
  - **`search_api.service.rrf_fuse`** — Reciprocal Rank Fusion skeleton → [open](vscode://file//home/paul/KGForge/src/search_api/service.py:6:1) | [view](service.py#L6-L11)
- **`search_api.splade_index`** — Module for search_api.splade_index → [open](vscode://file//home/paul/KGForge/src/search_api/splade_index.py:1:1) | [view](splade_index.py#L1)
  - **`search_api.splade_index.SpladeDoc`** — Spladedoc → [open](vscode://file//home/paul/KGForge/src/search_api/splade_index.py:26:1) | [view](splade_index.py#L26-L33)
  - **`search_api.splade_index.SpladeIndex`** — Spladeindex → [open](vscode://file//home/paul/KGForge/src/search_api/splade_index.py:36:1) | [view](splade_index.py#L36-L128)
  - **`search_api.splade_index.tok`** — Tok → [open](vscode://file//home/paul/KGForge/src/search_api/splade_index.py:14:1) | [view](splade_index.py#L14-L23)
