# search_client

Client abstractions for calling the kgfoundry Search API

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_client/__init__.py)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`, `search_client.client.KGFoundryClient`

## Neighborhood

```d2
direction: right
"search_client": "search_client" { link: "./search_client.md" }
"__future__.annotations": "__future__.annotations"
"search_client" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_client" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"search_client" -> "kgfoundry_common.navmap_types.NavMap"
"search_client.client.KGFoundryClient": "search_client.client.KGFoundryClient"
"search_client" -> "search_client.client.KGFoundryClient"
"search_client_code": "search_client code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_client/__init__.py" }
"search_client" -> "search_client_code" { style: dashed }
```

