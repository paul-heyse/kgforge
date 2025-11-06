# search_api.faiss_adapter

FastAPI service exposing search endpoints, aggregation helpers, and Problem Details responses.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/faiss_adapter.py)

## Sections

- **Public API**

## Contents

### search_api.faiss_adapter.DenseVecs

::: search_api.faiss_adapter.DenseVecs

### search_api.faiss_adapter.FaissAdapter

::: search_api.faiss_adapter.FaissAdapter

### search_api.faiss_adapter.FaissAdapterConfig

::: search_api.faiss_adapter.FaissAdapterConfig

### search_api.faiss_adapter._as_optional_str

::: search_api.faiss_adapter._as_optional_str

### search_api.faiss_adapter._is_faiss_index

::: search_api.faiss_adapter._is_faiss_index

### search_api.faiss_adapter._load_faiss_module

::: search_api.faiss_adapter._load_faiss_module

### search_api.faiss_adapter._load_libcuvs

::: search_api.faiss_adapter._load_libcuvs

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Callable`, `collections.abc.Sequence`, `dataclasses.dataclass`, `duckdb`, `importlib`, `kgfoundry_common.errors.IndexBuildError`, `kgfoundry_common.errors.VectorSearchError`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.numpy_typing.FloatMatrix`, `kgfoundry_common.numpy_typing.FloatVector`, `kgfoundry_common.numpy_typing.IntVector`, `kgfoundry_common.numpy_typing.normalize_l2`, `kgfoundry_common.numpy_typing.topk_indices`, `logging`, `numpy`, `numpy.typing`, `numpy.typing.NDArray`, `pathlib.Path`, `registry.duckdb_helpers.fetch_all`, `registry.duckdb_helpers.fetch_one`, `search_api.faiss_gpu.GpuContext`, `search_api.faiss_gpu.clone_index_to_gpu`, `search_api.faiss_gpu.configure_search_parameters`, `search_api.faiss_gpu.detect_gpu_context`, `search_api.types.FaissIndexProtocol`, `search_api.types.FaissModuleProtocol`, `typing.ClassVar`, `typing.Final`, `typing.TYPE_CHECKING`, `typing.TypeGuard`, `typing.cast`

**Imported by:** [search_api](./search_api.md)

## Autorefs Examples

- [search_api.faiss_adapter.DenseVecs][]
- [search_api.faiss_adapter.FaissAdapter][]
- [search_api.faiss_adapter.FaissAdapterConfig][]
- [search_api.faiss_adapter._as_optional_str][]
- [search_api.faiss_adapter._is_faiss_index][]
- [search_api.faiss_adapter._load_faiss_module][]

## Inheritance

```mermaid
classDiagram
    class DenseVecs
    class FaissAdapter
    class FaissAdapterConfig
```

## Neighborhood

```d2
direction: right
"search_api.faiss_adapter": "search_api.faiss_adapter" { link: "./search_api/faiss_adapter.md" }
"__future__.annotations": "__future__.annotations"
"search_api.faiss_adapter" -> "__future__.annotations"
"collections.abc.Callable": "collections.abc.Callable"
"search_api.faiss_adapter" -> "collections.abc.Callable"
"collections.abc.Sequence": "collections.abc.Sequence"
"search_api.faiss_adapter" -> "collections.abc.Sequence"
"dataclasses.dataclass": "dataclasses.dataclass"
"search_api.faiss_adapter" -> "dataclasses.dataclass"
"duckdb": "duckdb"
"search_api.faiss_adapter" -> "duckdb"
"importlib": "importlib"
"search_api.faiss_adapter" -> "importlib"
"kgfoundry_common.errors.IndexBuildError": "kgfoundry_common.errors.IndexBuildError"
"search_api.faiss_adapter" -> "kgfoundry_common.errors.IndexBuildError"
"kgfoundry_common.errors.VectorSearchError": "kgfoundry_common.errors.VectorSearchError"
"search_api.faiss_adapter" -> "kgfoundry_common.errors.VectorSearchError"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"search_api.faiss_adapter" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.numpy_typing.FloatMatrix": "kgfoundry_common.numpy_typing.FloatMatrix"
"search_api.faiss_adapter" -> "kgfoundry_common.numpy_typing.FloatMatrix"
"kgfoundry_common.numpy_typing.FloatVector": "kgfoundry_common.numpy_typing.FloatVector"
"search_api.faiss_adapter" -> "kgfoundry_common.numpy_typing.FloatVector"
"kgfoundry_common.numpy_typing.IntVector": "kgfoundry_common.numpy_typing.IntVector"
"search_api.faiss_adapter" -> "kgfoundry_common.numpy_typing.IntVector"
"kgfoundry_common.numpy_typing.normalize_l2": "kgfoundry_common.numpy_typing.normalize_l2"
"search_api.faiss_adapter" -> "kgfoundry_common.numpy_typing.normalize_l2"
"kgfoundry_common.numpy_typing.topk_indices": "kgfoundry_common.numpy_typing.topk_indices"
"search_api.faiss_adapter" -> "kgfoundry_common.numpy_typing.topk_indices"
"logging": "logging"
"search_api.faiss_adapter" -> "logging"
"numpy": "numpy"
"search_api.faiss_adapter" -> "numpy"
"numpy.typing": "numpy.typing"
"search_api.faiss_adapter" -> "numpy.typing"
"numpy.typing.NDArray": "numpy.typing.NDArray"
"search_api.faiss_adapter" -> "numpy.typing.NDArray"
"pathlib.Path": "pathlib.Path"
"search_api.faiss_adapter" -> "pathlib.Path"
"registry.duckdb_helpers.fetch_all": "registry.duckdb_helpers.fetch_all"
"search_api.faiss_adapter" -> "registry.duckdb_helpers.fetch_all"
"registry.duckdb_helpers.fetch_one": "registry.duckdb_helpers.fetch_one"
"search_api.faiss_adapter" -> "registry.duckdb_helpers.fetch_one"
"search_api.faiss_gpu.GpuContext": "search_api.faiss_gpu.GpuContext"
"search_api.faiss_adapter" -> "search_api.faiss_gpu.GpuContext"
"search_api.faiss_gpu.clone_index_to_gpu": "search_api.faiss_gpu.clone_index_to_gpu"
"search_api.faiss_adapter" -> "search_api.faiss_gpu.clone_index_to_gpu"
"search_api.faiss_gpu.configure_search_parameters": "search_api.faiss_gpu.configure_search_parameters"
"search_api.faiss_adapter" -> "search_api.faiss_gpu.configure_search_parameters"
"search_api.faiss_gpu.detect_gpu_context": "search_api.faiss_gpu.detect_gpu_context"
"search_api.faiss_adapter" -> "search_api.faiss_gpu.detect_gpu_context"
"search_api.types.FaissIndexProtocol": "search_api.types.FaissIndexProtocol"
"search_api.faiss_adapter" -> "search_api.types.FaissIndexProtocol"
"search_api.types.FaissModuleProtocol": "search_api.types.FaissModuleProtocol"
"search_api.faiss_adapter" -> "search_api.types.FaissModuleProtocol"
"typing.ClassVar": "typing.ClassVar"
"search_api.faiss_adapter" -> "typing.ClassVar"
"typing.Final": "typing.Final"
"search_api.faiss_adapter" -> "typing.Final"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"search_api.faiss_adapter" -> "typing.TYPE_CHECKING"
"typing.TypeGuard": "typing.TypeGuard"
"search_api.faiss_adapter" -> "typing.TypeGuard"
"typing.cast": "typing.cast"
"search_api.faiss_adapter" -> "typing.cast"
"search_api": "search_api" { link: "./search_api.md" }
"search_api" -> "search_api.faiss_adapter"
"search_api.faiss_adapter_code": "search_api.faiss_adapter code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/search_api/faiss_adapter.py" }
"search_api.faiss_adapter" -> "search_api.faiss_adapter_code" { style: dashed }
```

