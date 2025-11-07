# kgfoundry_common.gpu

Shared utilities and data structures used across KgFoundry services and tools.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/gpu.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)

## Sections

- **Public API**

## Contents

### kgfoundry_common.gpu._modules_available

::: kgfoundry_common.gpu._modules_available

### kgfoundry_common.gpu.has_gpu_stack

::: kgfoundry_common.gpu.has_gpu_stack

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Callable`, `collections.abc.Iterable`, `importlib`, `os`, `types.ModuleType`, `typing.TYPE_CHECKING`, `typing.cast`

## Autorefs Examples

- [kgfoundry_common.gpu._modules_available][]
- [kgfoundry_common.gpu.has_gpu_stack][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.gpu": "kgfoundry_common.gpu" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/gpu.py" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.gpu" -> "__future__.annotations"
"collections.abc.Callable": "collections.abc.Callable"
"kgfoundry_common.gpu" -> "collections.abc.Callable"
"collections.abc.Iterable": "collections.abc.Iterable"
"kgfoundry_common.gpu" -> "collections.abc.Iterable"
"importlib": "importlib"
"kgfoundry_common.gpu" -> "importlib"
"os": "os"
"kgfoundry_common.gpu" -> "os"
"types.ModuleType": "types.ModuleType"
"kgfoundry_common.gpu" -> "types.ModuleType"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"kgfoundry_common.gpu" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"kgfoundry_common.gpu" -> "typing.cast"
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.gpu" { style: dashed }
```

