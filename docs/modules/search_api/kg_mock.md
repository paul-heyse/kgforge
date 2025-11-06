# search_api.kg_mock

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/kg_mock.py)

## Sections

- **Public API**

## Contents

### search_api.kg_mock.ConceptMeta

::: search_api.kg_mock.ConceptMeta

*Bases:* TypedDict

### search_api.kg_mock.detect_query_concepts

::: search_api.kg_mock.detect_query_concepts

### search_api.kg_mock.kg_boost

::: search_api.kg_mock.kg_boost

### search_api.kg_mock.linked_concepts_for_text

::: search_api.kg_mock.linked_concepts_for_text

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `typing.Final`, `typing.TypedDict`

**Imported by:** [search_api](../search_api.md)

## Autorefs Examples

- [search_api.kg_mock.ConceptMeta][]
- [search_api.kg_mock.detect_query_concepts][]
- [search_api.kg_mock.kg_boost][]
- [search_api.kg_mock.linked_concepts_for_text][]

## Inheritance

```mermaid
classDiagram
    class ConceptMeta
    class TypedDict
    TypedDict <|-- ConceptMeta
```

## Neighborhood

```d2
direction: right
"search_api.kg_mock": "search_api.kg_mock" { link: "./search_api/kg_mock.md" }
"__future__.annotations": "__future__.annotations"
"search_api.kg_mock" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.kg_mock" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"typing.Final": "typing.Final"
"search_api.kg_mock" -> "typing.Final"
"typing.TypedDict": "typing.TypedDict"
"search_api.kg_mock" -> "typing.TypedDict"
"search_api": "search_api" { link: "./search_api.md" }
"search_api" -> "search_api.kg_mock"
"search_api.kg_mock_code": "search_api.kg_mock code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/kg_mock.py" }
"search_api.kg_mock" -> "search_api.kg_mock_code" { style: dashed }
```

