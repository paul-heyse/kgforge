# observability.metrics

Prometheus metrics shared across observability surfaces.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/metrics.py)

## Hierarchy

- **Parent:** [observability](../observability.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.prometheus.CounterLike`, `kgfoundry_common.prometheus.HistogramLike`, `kgfoundry_common.prometheus.build_counter`, `kgfoundry_common.prometheus.build_histogram`, `typing.TYPE_CHECKING`

## Neighborhood

```d2
direction: right
"observability.metrics": "observability.metrics" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/metrics.py" }
"__future__.annotations": "__future__.annotations"
"observability.metrics" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"observability.metrics" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.prometheus.CounterLike": "kgfoundry_common.prometheus.CounterLike"
"observability.metrics" -> "kgfoundry_common.prometheus.CounterLike"
"kgfoundry_common.prometheus.HistogramLike": "kgfoundry_common.prometheus.HistogramLike"
"observability.metrics" -> "kgfoundry_common.prometheus.HistogramLike"
"kgfoundry_common.prometheus.build_counter": "kgfoundry_common.prometheus.build_counter"
"observability.metrics" -> "kgfoundry_common.prometheus.build_counter"
"kgfoundry_common.prometheus.build_histogram": "kgfoundry_common.prometheus.build_histogram"
"observability.metrics" -> "kgfoundry_common.prometheus.build_histogram"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"observability.metrics" -> "typing.TYPE_CHECKING"
"observability": "observability" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/observability/__init__.py" }
"observability" -> "observability.metrics" { style: dashed }
```

