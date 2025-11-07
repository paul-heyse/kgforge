# search_client

Client abstractions for calling the kgfoundry Search API

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/search_client/__init__.py)

## Hierarchy

- **Children:** [search_client.client](search_client/client.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`, `search_client.client.KGFoundryClient`

## Neighborhood

```d2
direction: right
"search_client": "search_client" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_client/__init__.py" }
"__future__.annotations": "__future__.annotations"
"search_client" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_client" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"search_client" -> "kgfoundry_common.navmap_types.NavMap"
"search_client.client.KGFoundryClient": "search_client.client.KGFoundryClient"
"search_client" -> "search_client.client.KGFoundryClient"
"search_client.client": "search_client.client" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_client/client.py" }
"search_client" -> "search_client.client" { style: dashed }
```

