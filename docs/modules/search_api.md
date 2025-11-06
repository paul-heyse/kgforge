# search_api

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/__init__.py)

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

**Imports:** `__future__.annotations`, `importlib.import_module`, `kgfoundry_common.navmap_loader.load_nav_metadata`, [search_api.app](./search_api/app.md), [search_api.bm25_index](./search_api/bm25_index.md), [search_api.faiss_adapter](./search_api/faiss_adapter.md), [search_api.fixture_index](./search_api/fixture_index.md), [search_api.fusion](./search_api/fusion.md), [search_api.kg_mock](./search_api/kg_mock.md), [search_api.schemas](./search_api/schemas.md), [search_api.service](./search_api/service.md), [search_api.splade_index](./search_api/splade_index.md), [search_api.types](./search_api/types.md), `sys`, `types.ModuleType`, `typing.TYPE_CHECKING`

## Autorefs Examples

- [search_api.__dir__][]
- [search_api.__getattr__][]
- [search_api._load][]

## Neighborhood

```d2
direction: right
"search_api": "search_api" { link: "./search_api.md" }
"__future__.annotations": "__future__.annotations"
"search_api" -> "__future__.annotations"
"importlib.import_module": "importlib.import_module"
"search_api" -> "importlib.import_module"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.app": "search_api.app" { link: "./search_api/app.md" }
"search_api" -> "search_api.app"
"search_api.bm25_index": "search_api.bm25_index" { link: "./search_api/bm25_index.md" }
"search_api" -> "search_api.bm25_index"
"search_api.faiss_adapter": "search_api.faiss_adapter" { link: "./search_api/faiss_adapter.md" }
"search_api" -> "search_api.faiss_adapter"
"search_api.fixture_index": "search_api.fixture_index" { link: "./search_api/fixture_index.md" }
"search_api" -> "search_api.fixture_index"
"search_api.fusion": "search_api.fusion" { link: "./search_api/fusion.md" }
"search_api" -> "search_api.fusion"
"search_api.kg_mock": "search_api.kg_mock" { link: "./search_api/kg_mock.md" }
"search_api" -> "search_api.kg_mock"
"search_api.schemas": "search_api.schemas" { link: "./search_api/schemas.md" }
"search_api" -> "search_api.schemas"
"search_api.service": "search_api.service" { link: "./search_api/service.md" }
"search_api" -> "search_api.service"
"search_api.splade_index": "search_api.splade_index" { link: "./search_api/splade_index.md" }
"search_api" -> "search_api.splade_index"
"search_api.types": "search_api.types" { link: "./search_api/types.md" }
"search_api" -> "search_api.types"
"sys": "sys"
"search_api" -> "sys"
"types.ModuleType": "types.ModuleType"
"search_api" -> "types.ModuleType"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api" -> "typing.TYPE_CHECKING"
"search_api_code": "search_api code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/__init__.py" }
"search_api" -> "search_api_code" { style: dashed }
```

