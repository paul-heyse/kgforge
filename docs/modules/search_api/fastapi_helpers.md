# search_api.fastapi_helpers

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fastapi_helpers.py)

## Hierarchy

- **Parent:** [search_api](../search_api.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.fastapi_helpers.DEFAULT_TIMEOUT_SECONDS`, `kgfoundry_common.fastapi_helpers.typed_dependency`, `kgfoundry_common.fastapi_helpers.typed_exception_handler`, `kgfoundry_common.fastapi_helpers.typed_middleware`, `kgfoundry_common.navmap_loader.load_nav_metadata`

## Neighborhood

```d2
direction: right
"search_api.fastapi_helpers": "search_api.fastapi_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fastapi_helpers.py" }
"__future__.annotations": "__future__.annotations"
"search_api.fastapi_helpers" -> "__future__.annotations"
"kgfoundry_common.fastapi_helpers.DEFAULT_TIMEOUT_SECONDS": "kgfoundry_common.fastapi_helpers.DEFAULT_TIMEOUT_SECONDS"
"search_api.fastapi_helpers" -> "kgfoundry_common.fastapi_helpers.DEFAULT_TIMEOUT_SECONDS"
"kgfoundry_common.fastapi_helpers.typed_dependency": "kgfoundry_common.fastapi_helpers.typed_dependency"
"search_api.fastapi_helpers" -> "kgfoundry_common.fastapi_helpers.typed_dependency"
"kgfoundry_common.fastapi_helpers.typed_exception_handler": "kgfoundry_common.fastapi_helpers.typed_exception_handler"
"search_api.fastapi_helpers" -> "kgfoundry_common.fastapi_helpers.typed_exception_handler"
"kgfoundry_common.fastapi_helpers.typed_middleware": "kgfoundry_common.fastapi_helpers.typed_middleware"
"search_api.fastapi_helpers" -> "kgfoundry_common.fastapi_helpers.typed_middleware"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.fastapi_helpers" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api": "search_api" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/__init__.py" }
"search_api" -> "search_api.fastapi_helpers" { style: dashed }
```

