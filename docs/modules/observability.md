# observability

Observability metrics and instrumentation helpers

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/__init__.py)

## Hierarchy

- **Children:** [observability.metrics](observability/metrics.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"observability": "observability" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/__init__.py" }
"__future__.annotations": "__future__.annotations"
"observability" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"observability" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"observability" -> "kgfoundry_common.navmap_types.NavMap"
"observability.metrics": "observability.metrics" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/metrics.py" }
"observability" -> "observability.metrics" { style: dashed }
```

