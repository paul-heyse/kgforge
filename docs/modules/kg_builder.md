# kg_builder

Knowledge graph builder components and interfaces

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/__init__.py)

## Hierarchy

- **Children:** [kg_builder.mock_kg](kg_builder/mock_kg.md), [kg_builder.neo4j_store](kg_builder/neo4j_store.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"kg_builder": "kg_builder" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/__init__.py" }
"__future__.annotations": "__future__.annotations"
"kg_builder" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kg_builder" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"kg_builder" -> "kgfoundry_common.navmap_types.NavMap"
"kg_builder.mock_kg": "kg_builder.mock_kg" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/mock_kg.py" }
"kg_builder" -> "kg_builder.mock_kg" { style: dashed }
"kg_builder.neo4j_store": "kg_builder.neo4j_store" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kg_builder/neo4j_store.py" }
"kg_builder" -> "kg_builder.neo4j_store" { style: dashed }
```

