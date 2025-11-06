# vectorstore_faiss.gpu

GPU-aware FAISS index helpers backed by the shared search API facade.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/vectorstore_faiss/gpu.py)

## Sections

- **Public API**

## Contents

### vectorstore_faiss.gpu.FaissGpuIndex

::: vectorstore_faiss.gpu.FaissGpuIndex

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Sequence`, `dataclasses.dataclass`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.numpy_typing.FloatMatrix`, `kgfoundry_common.numpy_typing.FloatVector`, `kgfoundry_common.numpy_typing.IntVector`, `kgfoundry_common.numpy_typing.normalize_l2`, `logging`, `numpy`, `numpy.typing`, `search_api.faiss_gpu.GpuContext`, `search_api.faiss_gpu.clone_index_to_gpu`, `search_api.faiss_gpu.configure_search_parameters`, `search_api.faiss_gpu.detect_gpu_context`, `search_api.types.FaissIndexProtocol`, `search_api.types.FaissModuleProtocol`, `typing.TYPE_CHECKING`, `typing.cast`

## Autorefs Examples

- [vectorstore_faiss.gpu.FaissGpuIndex][]

## Inheritance

```mermaid
classDiagram
    class FaissGpuIndex
```

## Neighborhood

```d2
direction: right
"vectorstore_faiss.gpu": "vectorstore_faiss.gpu" { link: "./vectorstore_faiss/gpu.md" }
"__future__.annotations": "__future__.annotations"
"vectorstore_faiss.gpu" -> "__future__.annotations"
"collections.abc.Sequence": "collections.abc.Sequence"
"vectorstore_faiss.gpu" -> "collections.abc.Sequence"
"dataclasses.dataclass": "dataclasses.dataclass"
"vectorstore_faiss.gpu" -> "dataclasses.dataclass"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"vectorstore_faiss.gpu" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.numpy_typing.FloatMatrix": "kgfoundry_common.numpy_typing.FloatMatrix"
"vectorstore_faiss.gpu" -> "kgfoundry_common.numpy_typing.FloatMatrix"
"kgfoundry_common.numpy_typing.FloatVector": "kgfoundry_common.numpy_typing.FloatVector"
"vectorstore_faiss.gpu" -> "kgfoundry_common.numpy_typing.FloatVector"
"kgfoundry_common.numpy_typing.IntVector": "kgfoundry_common.numpy_typing.IntVector"
"vectorstore_faiss.gpu" -> "kgfoundry_common.numpy_typing.IntVector"
"kgfoundry_common.numpy_typing.normalize_l2": "kgfoundry_common.numpy_typing.normalize_l2"
"vectorstore_faiss.gpu" -> "kgfoundry_common.numpy_typing.normalize_l2"
"logging": "logging"
"vectorstore_faiss.gpu" -> "logging"
"numpy": "numpy"
"vectorstore_faiss.gpu" -> "numpy"
"numpy.typing": "numpy.typing"
"vectorstore_faiss.gpu" -> "numpy.typing"
"search_api.faiss_gpu.GpuContext": "search_api.faiss_gpu.GpuContext"
"vectorstore_faiss.gpu" -> "search_api.faiss_gpu.GpuContext"
"search_api.faiss_gpu.clone_index_to_gpu": "search_api.faiss_gpu.clone_index_to_gpu"
"vectorstore_faiss.gpu" -> "search_api.faiss_gpu.clone_index_to_gpu"
"search_api.faiss_gpu.configure_search_parameters": "search_api.faiss_gpu.configure_search_parameters"
"vectorstore_faiss.gpu" -> "search_api.faiss_gpu.configure_search_parameters"
"search_api.faiss_gpu.detect_gpu_context": "search_api.faiss_gpu.detect_gpu_context"
"vectorstore_faiss.gpu" -> "search_api.faiss_gpu.detect_gpu_context"
"search_api.types.FaissIndexProtocol": "search_api.types.FaissIndexProtocol"
"vectorstore_faiss.gpu" -> "search_api.types.FaissIndexProtocol"
"search_api.types.FaissModuleProtocol": "search_api.types.FaissModuleProtocol"
"vectorstore_faiss.gpu" -> "search_api.types.FaissModuleProtocol"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"vectorstore_faiss.gpu" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"vectorstore_faiss.gpu" -> "typing.cast"
"vectorstore_faiss.gpu_code": "vectorstore_faiss.gpu code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/vectorstore_faiss/gpu.py" }
"vectorstore_faiss.gpu" -> "vectorstore_faiss.gpu_code" { style: dashed }
```

