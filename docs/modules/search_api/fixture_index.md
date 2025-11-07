# search_api.fixture_index

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fixture_index.py)

## Hierarchy

- **Parent:** [search_api](../search_api.md)

## Sections

- **Public API**

## Contents

### search_api.fixture_index.FixtureDoc

::: search_api.fixture_index.FixtureDoc

### search_api.fixture_index.FixtureIndex

::: search_api.fixture_index.FixtureIndex

### search_api.fixture_index._as_str

::: search_api.fixture_index._as_str

### search_api.fixture_index.tokenize

::: search_api.fixture_index.tokenize

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Iterator`, `collections.abc.Sequence`, `dataclasses.dataclass`, `duckdb`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `math`, `pathlib.Path`, `re`, `registry.duckdb_helpers.fetch_all`, `registry.duckdb_helpers.fetch_one`, `typing.TYPE_CHECKING`

**Imported by:** [search_api](../search_api.md)

## Autorefs Examples

- [search_api.fixture_index.FixtureDoc][]
- [search_api.fixture_index.FixtureIndex][]
- [search_api.fixture_index._as_str][]
- [search_api.fixture_index.tokenize][]

## Inheritance

```mermaid
classDiagram
    class FixtureDoc
    class FixtureIndex
```

## Neighborhood

```d2
direction: right
"search_api.fixture_index": "search_api.fixture_index" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/fixture_index.py" }
"__future__.annotations": "__future__.annotations"
"search_api.fixture_index" -> "__future__.annotations"
"collections.abc.Iterator": "collections.abc.Iterator"
"search_api.fixture_index" -> "collections.abc.Iterator"
"collections.abc.Sequence": "collections.abc.Sequence"
"search_api.fixture_index" -> "collections.abc.Sequence"
"dataclasses.dataclass": "dataclasses.dataclass"
"search_api.fixture_index" -> "dataclasses.dataclass"
"duckdb": "duckdb"
"search_api.fixture_index" -> "duckdb"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.fixture_index" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"math": "math"
"search_api.fixture_index" -> "math"
"pathlib.Path": "pathlib.Path"
"search_api.fixture_index" -> "pathlib.Path"
"re": "re"
"search_api.fixture_index" -> "re"
"registry.duckdb_helpers.fetch_all": "registry.duckdb_helpers.fetch_all"
"search_api.fixture_index" -> "registry.duckdb_helpers.fetch_all"
"registry.duckdb_helpers.fetch_one": "registry.duckdb_helpers.fetch_one"
"search_api.fixture_index" -> "registry.duckdb_helpers.fetch_one"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api.fixture_index" -> "typing.TYPE_CHECKING"
"search_api": "search_api" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/search_api/__init__.py" }
"search_api" -> "search_api.fixture_index"
"search_api" -> "search_api.fixture_index" { style: dashed }
```

