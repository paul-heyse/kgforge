# search_api.service

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

## Sections

- **Public API**

## Contents

### search_api.service.apply_kg_boosts

::: search_api.service.apply_kg_boosts

### search_api.service.mmr_deduplicate

::: search_api.service.mmr_deduplicate

### search_api.service.rrf_fuse

::: search_api.service.rrf_fuse

### search_api.service.search_service

::: search_api.service.search_service

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Mapping`, `kgfoundry_common.errors.exceptions.VectorSearchError`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.logging.with_fields`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.observability.MetricsProvider`, `kgfoundry_common.observability.observe_duration`, `kgfoundry_common.problem_details.JsonValue`, `search_api.types.AgentSearchResponse`, `search_api.types.VectorSearchResultTypedDict`, `time`, `typing.TYPE_CHECKING`

**Imported by:** [search_api](../search_api.md)

## Autorefs Examples

- [search_api.service.apply_kg_boosts][]
- [search_api.service.mmr_deduplicate][]
- [search_api.service.rrf_fuse][]

## Neighborhood

```d2
direction: right
"search_api.service": "search_api.service" { link: "service.md" }
"__future__.annotations": "__future__.annotations"
"search_api.service" -> "__future__.annotations"
"collections.abc.Mapping": "collections.abc.Mapping"
"search_api.service" -> "collections.abc.Mapping"
"kgfoundry_common.errors.exceptions.VectorSearchError": "kgfoundry_common.errors.exceptions.VectorSearchError"
"search_api.service" -> "kgfoundry_common.errors.exceptions.VectorSearchError"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"search_api.service" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.logging.with_fields": "kgfoundry_common.logging.with_fields"
"search_api.service" -> "kgfoundry_common.logging.with_fields"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.service" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.observability.MetricsProvider": "kgfoundry_common.observability.MetricsProvider"
"search_api.service" -> "kgfoundry_common.observability.MetricsProvider"
"kgfoundry_common.observability.observe_duration": "kgfoundry_common.observability.observe_duration"
"search_api.service" -> "kgfoundry_common.observability.observe_duration"
"kgfoundry_common.problem_details.JsonValue": "kgfoundry_common.problem_details.JsonValue"
"search_api.service" -> "kgfoundry_common.problem_details.JsonValue"
"search_api.types.AgentSearchResponse": "search_api.types.AgentSearchResponse"
"search_api.service" -> "search_api.types.AgentSearchResponse"
"search_api.types.VectorSearchResultTypedDict": "search_api.types.VectorSearchResultTypedDict"
"search_api.service" -> "search_api.types.VectorSearchResultTypedDict"
"time": "time"
"search_api.service" -> "time"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api.service" -> "typing.TYPE_CHECKING"
"search_api": "search_api" { link: "../search_api.md" }
"search_api" -> "search_api.service"
```

