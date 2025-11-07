# kg_builder.neo4j_store

Placeholder interface for a Neo4j-backed store

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/neo4j_store.py)

## Hierarchy

- **Parent:** [kg_builder](../kg_builder.md)

## Sections

- **Public API**

## Contents

### kg_builder.neo4j_store.Neo4jStore

::: kg_builder.neo4j_store.Neo4jStore

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`

## Autorefs Examples

- [kg_builder.neo4j_store.Neo4jStore][]

## Inheritance

```mermaid
classDiagram
    class Neo4jStore
```

## Neighborhood

```d2
direction: right
"kg_builder.neo4j_store": "kg_builder.neo4j_store" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/neo4j_store.py" }
"__future__.annotations": "__future__.annotations"
"kg_builder.neo4j_store" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kg_builder.neo4j_store" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kg_builder": "kg_builder" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/__init__.py" }
"kg_builder" -> "kg_builder.neo4j_store" { style: dashed }
```

