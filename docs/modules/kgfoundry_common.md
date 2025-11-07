# kgfoundry_common

Shared utilities and data structures used across KgFoundry services and tools.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py)

## Hierarchy

- **Children:** [kgfoundry_common.config](kgfoundry_common/config.md), [kgfoundry_common.errors](kgfoundry_common/errors.md), [kgfoundry_common.exceptions](kgfoundry_common/exceptions.md), [kgfoundry_common.fastapi_helpers](kgfoundry_common/fastapi_helpers.md), [kgfoundry_common.fs](kgfoundry_common/fs.md), [kgfoundry_common.gpu](kgfoundry_common/gpu.md), [kgfoundry_common.http](kgfoundry_common/http.md), [kgfoundry_common.ids](kgfoundry_common/ids.md), [kgfoundry_common.jsonschema_utils](kgfoundry_common/jsonschema_utils.md), [kgfoundry_common.logging](kgfoundry_common/logging.md), [kgfoundry_common.models](kgfoundry_common/models.md), [kgfoundry_common.navmap_loader](kgfoundry_common/navmap_loader.md), [kgfoundry_common.navmap_types](kgfoundry_common/navmap_types.md), [kgfoundry_common.numpy_typing](kgfoundry_common/numpy_typing.md), [kgfoundry_common.observability](kgfoundry_common/observability.md), [kgfoundry_common.opentelemetry_types](kgfoundry_common/opentelemetry_types.md), [kgfoundry_common.optional_deps](kgfoundry_common/optional_deps.md), [kgfoundry_common.parquet_io](kgfoundry_common/parquet_io.md), [kgfoundry_common.problem_details](kgfoundry_common/problem_details.md), [kgfoundry_common.prometheus](kgfoundry_common/prometheus.md), [kgfoundry_common.pydantic](kgfoundry_common/pydantic.md), [kgfoundry_common.safe_pickle_v2](kgfoundry_common/safe_pickle_v2.md), [kgfoundry_common.schema_helpers](kgfoundry_common/schema_helpers.md), [kgfoundry_common.sequence_guards](kgfoundry_common/sequence_guards.md), [kgfoundry_common.serialization](kgfoundry_common/serialization.md), [kgfoundry_common.settings](kgfoundry_common/settings.md), [kgfoundry_common.subprocess_utils](kgfoundry_common/subprocess_utils.md), [kgfoundry_common.types](kgfoundry_common/types.md), [kgfoundry_common.typing](kgfoundry_common/typing.md), [kgfoundry_common.vector_types](kgfoundry_common/vector_types.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`

## Neighborhood

```d2
direction: right
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.config": "kgfoundry_common.config" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/config.py" }
"kgfoundry_common" -> "kgfoundry_common.config" { style: dashed }
"kgfoundry_common.errors": "kgfoundry_common.errors" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/errors/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.errors" { style: dashed }
"kgfoundry_common.exceptions": "kgfoundry_common.exceptions" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/exceptions.py" }
"kgfoundry_common" -> "kgfoundry_common.exceptions" { style: dashed }
"kgfoundry_common.fastapi_helpers": "kgfoundry_common.fastapi_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/fastapi_helpers.py" }
"kgfoundry_common" -> "kgfoundry_common.fastapi_helpers" { style: dashed }
"kgfoundry_common.fs": "kgfoundry_common.fs" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/fs.py" }
"kgfoundry_common" -> "kgfoundry_common.fs" { style: dashed }
"kgfoundry_common.gpu": "kgfoundry_common.gpu" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/gpu.py" }
"kgfoundry_common" -> "kgfoundry_common.gpu" { style: dashed }
"kgfoundry_common.http": "kgfoundry_common.http" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/http/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.http" { style: dashed }
"kgfoundry_common.ids": "kgfoundry_common.ids" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/ids.py" }
"kgfoundry_common" -> "kgfoundry_common.ids" { style: dashed }
"kgfoundry_common.jsonschema_utils": "kgfoundry_common.jsonschema_utils" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/jsonschema_utils.py" }
"kgfoundry_common" -> "kgfoundry_common.jsonschema_utils" { style: dashed }
"kgfoundry_common.logging": "kgfoundry_common.logging" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/logging.py" }
"kgfoundry_common" -> "kgfoundry_common.logging" { style: dashed }
"kgfoundry_common.models": "kgfoundry_common.models" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/models.py" }
"kgfoundry_common" -> "kgfoundry_common.models" { style: dashed }
"kgfoundry_common.navmap_loader": "kgfoundry_common.navmap_loader" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/navmap_loader.py" }
"kgfoundry_common" -> "kgfoundry_common.navmap_loader" { style: dashed }
"kgfoundry_common.navmap_types": "kgfoundry_common.navmap_types" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/navmap_types.py" }
"kgfoundry_common" -> "kgfoundry_common.navmap_types" { style: dashed }
"kgfoundry_common.numpy_typing": "kgfoundry_common.numpy_typing" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/numpy_typing.py" }
"kgfoundry_common" -> "kgfoundry_common.numpy_typing" { style: dashed }
"kgfoundry_common.observability": "kgfoundry_common.observability" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/observability.py" }
"kgfoundry_common" -> "kgfoundry_common.observability" { style: dashed }
"kgfoundry_common.opentelemetry_types": "kgfoundry_common.opentelemetry_types" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/opentelemetry_types.py" }
"kgfoundry_common" -> "kgfoundry_common.opentelemetry_types" { style: dashed }
"kgfoundry_common.optional_deps": "kgfoundry_common.optional_deps" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/optional_deps.py" }
"kgfoundry_common" -> "kgfoundry_common.optional_deps" { style: dashed }
"kgfoundry_common.parquet_io": "kgfoundry_common.parquet_io" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/parquet_io.py" }
"kgfoundry_common" -> "kgfoundry_common.parquet_io" { style: dashed }
"kgfoundry_common.problem_details": "kgfoundry_common.problem_details" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/problem_details.py" }
"kgfoundry_common" -> "kgfoundry_common.problem_details" { style: dashed }
"kgfoundry_common.prometheus": "kgfoundry_common.prometheus" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/prometheus.py" }
"kgfoundry_common" -> "kgfoundry_common.prometheus" { style: dashed }
"kgfoundry_common.pydantic": "kgfoundry_common.pydantic" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/pydantic.py" }
"kgfoundry_common" -> "kgfoundry_common.pydantic" { style: dashed }
"kgfoundry_common.safe_pickle_v2": "kgfoundry_common.safe_pickle_v2" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/safe_pickle_v2.py" }
"kgfoundry_common" -> "kgfoundry_common.safe_pickle_v2" { style: dashed }
"kgfoundry_common.schema_helpers": "kgfoundry_common.schema_helpers" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/schema_helpers.py" }
"kgfoundry_common" -> "kgfoundry_common.schema_helpers" { style: dashed }
"kgfoundry_common.sequence_guards": "kgfoundry_common.sequence_guards" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/sequence_guards.py" }
"kgfoundry_common" -> "kgfoundry_common.sequence_guards" { style: dashed }
"kgfoundry_common.serialization": "kgfoundry_common.serialization" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/serialization.py" }
"kgfoundry_common" -> "kgfoundry_common.serialization" { style: dashed }
"kgfoundry_common.settings": "kgfoundry_common.settings" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/settings.py" }
"kgfoundry_common" -> "kgfoundry_common.settings" { style: dashed }
"kgfoundry_common.subprocess_utils": "kgfoundry_common.subprocess_utils" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/subprocess_utils.py" }
"kgfoundry_common" -> "kgfoundry_common.subprocess_utils" { style: dashed }
"kgfoundry_common.types": "kgfoundry_common.types" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/types.py" }
"kgfoundry_common" -> "kgfoundry_common.types" { style: dashed }
"kgfoundry_common.typing": "kgfoundry_common.typing" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/typing/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.typing" { style: dashed }
"kgfoundry_common.vector_types": "kgfoundry_common.vector_types" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/vector_types.py" }
"kgfoundry_common" -> "kgfoundry_common.vector_types" { style: dashed }
```

