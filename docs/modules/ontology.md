# ontology

Ontology loading and lookup helpers

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/ontology/__init__.py)

## Hierarchy

- **Children:** [ontology.catalog](ontology/catalog.md), [ontology.loader](ontology/loader.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"ontology": "ontology" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/ontology/__init__.py" }
"__future__.annotations": "__future__.annotations"
"ontology" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"ontology" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"ontology" -> "kgfoundry_common.navmap_types.NavMap"
"ontology.catalog": "ontology.catalog" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/ontology/catalog.py" }
"ontology" -> "ontology.catalog" { style: dashed }
"ontology.loader": "ontology.loader" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/ontology/loader.py" }
"ontology" -> "ontology.loader" { style: dashed }
```

