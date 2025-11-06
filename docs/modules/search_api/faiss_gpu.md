# search_api.faiss_gpu

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/faiss_gpu.py)

## Sections

- **Public API**

## Contents

### search_api.faiss_gpu.GpuContext

::: search_api.faiss_gpu.GpuContext

### search_api.faiss_gpu.clone_index_to_gpu

::: search_api.faiss_gpu.clone_index_to_gpu

### search_api.faiss_gpu.configure_search_parameters

::: search_api.faiss_gpu.configure_search_parameters

### search_api.faiss_gpu.detect_gpu_context

::: search_api.faiss_gpu.detect_gpu_context

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Callable`, `collections.abc.Sequence`, `dataclasses.dataclass`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.sequence_guards.first_or_error`, `kgfoundry_common.sequence_guards.first_or_error_multi_device`, `logging`, `search_api.types.FaissIndexProtocol`, `search_api.types.FaissModuleProtocol`, `search_api.types.GpuClonerOptionsProtocol`, `search_api.types.GpuResourcesProtocol`, `typing.TYPE_CHECKING`, `typing.cast`

## Autorefs Examples

- [search_api.faiss_gpu.GpuContext][]
- [search_api.faiss_gpu.clone_index_to_gpu][]
- [search_api.faiss_gpu.configure_search_parameters][]
- [search_api.faiss_gpu.detect_gpu_context][]

## Inheritance

```mermaid
classDiagram
    class GpuContext
```

## Neighborhood

```d2
direction: right
"search_api.faiss_gpu": "search_api.faiss_gpu" { link: "./search_api/faiss_gpu.md" }
"__future__.annotations": "__future__.annotations"
"search_api.faiss_gpu" -> "__future__.annotations"
"collections.abc.Callable": "collections.abc.Callable"
"search_api.faiss_gpu" -> "collections.abc.Callable"
"collections.abc.Sequence": "collections.abc.Sequence"
"search_api.faiss_gpu" -> "collections.abc.Sequence"
"dataclasses.dataclass": "dataclasses.dataclass"
"search_api.faiss_gpu" -> "dataclasses.dataclass"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.faiss_gpu" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.sequence_guards.first_or_error": "kgfoundry_common.sequence_guards.first_or_error"
"search_api.faiss_gpu" -> "kgfoundry_common.sequence_guards.first_or_error"
"kgfoundry_common.sequence_guards.first_or_error_multi_device": "kgfoundry_common.sequence_guards.first_or_error_multi_device"
"search_api.faiss_gpu" -> "kgfoundry_common.sequence_guards.first_or_error_multi_device"
"logging": "logging"
"search_api.faiss_gpu" -> "logging"
"search_api.types.FaissIndexProtocol": "search_api.types.FaissIndexProtocol"
"search_api.faiss_gpu" -> "search_api.types.FaissIndexProtocol"
"search_api.types.FaissModuleProtocol": "search_api.types.FaissModuleProtocol"
"search_api.faiss_gpu" -> "search_api.types.FaissModuleProtocol"
"search_api.types.GpuClonerOptionsProtocol": "search_api.types.GpuClonerOptionsProtocol"
"search_api.faiss_gpu" -> "search_api.types.GpuClonerOptionsProtocol"
"search_api.types.GpuResourcesProtocol": "search_api.types.GpuResourcesProtocol"
"search_api.faiss_gpu" -> "search_api.types.GpuResourcesProtocol"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api.faiss_gpu" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"search_api.faiss_gpu" -> "typing.cast"
"search_api.faiss_gpu_code": "search_api.faiss_gpu code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/faiss_gpu.py" }
"search_api.faiss_gpu" -> "search_api.faiss_gpu_code" { style: dashed }
```

