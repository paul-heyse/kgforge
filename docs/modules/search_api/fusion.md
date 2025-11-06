# search_api.fusion

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/fusion.py)

## Sections

- **Public API**

## Contents

### search_api.fusion.rrf_fuse

::: search_api.fusion.rrf_fuse

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`

**Imported by:** [search_api](./search_api.md)

## Autorefs Examples

- [search_api.fusion.rrf_fuse][]

## Neighborhood

```d2
direction: right
"search_api.fusion": "search_api.fusion" { link: "./search_api/fusion.md" }
"__future__.annotations": "__future__.annotations"
"search_api.fusion" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.fusion" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api": "search_api" { link: "./search_api.md" }
"search_api" -> "search_api.fusion"
"search_api.fusion_code": "search_api.fusion code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/fusion.py" }
"search_api.fusion" -> "search_api.fusion_code" { style: dashed }
```

