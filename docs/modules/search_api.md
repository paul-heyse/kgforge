# search_api

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/__init__.py)

## Hierarchy

- **Children:** [search_api.app](search_api/app.md), [search_api.bm25_index](search_api/bm25_index.md), [search_api.faiss_adapter](search_api/faiss_adapter.md), [search_api.faiss_gpu](search_api/faiss_gpu.md), [search_api.fastapi_helpers](search_api/fastapi_helpers.md), [search_api.fixture_index](search_api/fixture_index.md), [search_api.fusion](search_api/fusion.md), [search_api.kg_mock](search_api/kg_mock.md), [search_api.schemas](search_api/schemas.md), [search_api.service](search_api/service.md), [search_api.splade_index](search_api/splade_index.md), [search_api.types](search_api/types.md), [search_api.vectorstore_factory](search_api/vectorstore_factory.md)

## Sections

- **Public API**

## Contents

### search_api.__dir__

::: search_api.__dir__

### search_api.__getattr__

::: search_api.__getattr__

### search_api._load

::: search_api._load

## Relationships

**Imports:** `__future__.annotations`, `importlib.import_module`, `kgfoundry_common.navmap_loader.load_nav_metadata`, [search_api.app](search_api/app.md), [search_api.bm25_index](search_api/bm25_index.md), [search_api.faiss_adapter](search_api/faiss_adapter.md), [search_api.fixture_index](search_api/fixture_index.md), [search_api.fusion](search_api/fusion.md), [search_api.kg_mock](search_api/kg_mock.md), [search_api.schemas](search_api/schemas.md), [search_api.service](search_api/service.md), [search_api.splade_index](search_api/splade_index.md), [search_api.types](search_api/types.md), `sys`, `types.ModuleType`, `typing.TYPE_CHECKING`

## Autorefs Examples

- [search_api.__dir__][]
- [search_api.__getattr__][]
- [search_api._load][]

## Neighborhood

```d2
direction: right
"search_api": "search_api" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/__init__.py" }
"__future__.annotations": "__future__.annotations"
"search_api" -> "__future__.annotations"
"importlib.import_module": "importlib.import_module"
"search_api" -> "importlib.import_module"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.app": "search_api.app" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/app.py" }
"search_api" -> "search_api.app"
"search_api.bm25_index": "search_api.bm25_index" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/bm25_index.py" }
"search_api" -> "search_api.bm25_index"
"search_api.faiss_adapter": "search_api.faiss_adapter" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/faiss_adapter.py" }
"search_api" -> "search_api.faiss_adapter"
"search_api.fixture_index": "search_api.fixture_index" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fixture_index.py" }
"search_api" -> "search_api.fixture_index"
"search_api.fusion": "search_api.fusion" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fusion.py" }
"search_api" -> "search_api.fusion"
"search_api.kg_mock": "search_api.kg_mock" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/kg_mock.py" }
"search_api" -> "search_api.kg_mock"
"search_api.schemas": "search_api.schemas" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/schemas.py" }
"search_api" -> "search_api.schemas"
"search_api.service": "search_api.service" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/service.py" }
"search_api" -> "search_api.service"
"search_api.splade_index": "search_api.splade_index" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/splade_index.py" }
"search_api" -> "search_api.splade_index"
"search_api.types": "search_api.types" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/types.py" }
"search_api" -> "search_api.types"
"sys": "sys"
"search_api" -> "sys"
"types.ModuleType": "types.ModuleType"
"search_api" -> "types.ModuleType"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api" -> "typing.TYPE_CHECKING"
"search_api" -> "search_api.app" { style: dashed }
"search_api" -> "search_api.bm25_index" { style: dashed }
"search_api" -> "search_api.faiss_adapter" { style: dashed }
"search_api.faiss_gpu": "search_api.faiss_gpu" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/faiss_gpu.py" }
"search_api" -> "search_api.faiss_gpu" { style: dashed }
"search_api.fastapi_helpers": "search_api.fastapi_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fastapi_helpers.py" }
"search_api" -> "search_api.fastapi_helpers" { style: dashed }
"search_api" -> "search_api.fixture_index" { style: dashed }
"search_api" -> "search_api.fusion" { style: dashed }
"search_api" -> "search_api.kg_mock" { style: dashed }
"search_api" -> "search_api.schemas" { style: dashed }
"search_api" -> "search_api.service" { style: dashed }
"search_api" -> "search_api.splade_index" { style: dashed }
"search_api" -> "search_api.types" { style: dashed }
"search_api.vectorstore_factory": "search_api.vectorstore_factory" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/vectorstore_factory.py" }
"search_api" -> "search_api.vectorstore_factory" { style: dashed }
```

