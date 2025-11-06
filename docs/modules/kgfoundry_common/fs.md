# kgfoundry_common.fs

Shared utilities and data structures used across KgFoundry services and tools.

## Sections

- **Public API**

## Contents

### kgfoundry_common.fs.atomic_write

::: kgfoundry_common.fs.atomic_write

### kgfoundry_common.fs.ensure_dir

::: kgfoundry_common.fs.ensure_dir

### kgfoundry_common.fs.read_text

::: kgfoundry_common.fs.read_text

### kgfoundry_common.fs.safe_join

::: kgfoundry_common.fs.safe_join

### kgfoundry_common.fs.write_text

::: kgfoundry_common.fs.write_text

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.logging.get_logger`, `pathlib.Path`, `sys`, `tempfile`, `typing.Literal`

## Autorefs Examples

- [kgfoundry_common.fs.atomic_write][]
- [kgfoundry_common.fs.ensure_dir][]
- [kgfoundry_common.fs.read_text][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.fs": "kgfoundry_common.fs" { link: "fs.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.fs" -> "__future__.annotations"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.fs" -> "kgfoundry_common.logging.get_logger"
"pathlib.Path": "pathlib.Path"
"kgfoundry_common.fs" -> "pathlib.Path"
"sys": "sys"
"kgfoundry_common.fs" -> "sys"
"tempfile": "tempfile"
"kgfoundry_common.fs" -> "tempfile"
"typing.Literal": "typing.Literal"
"kgfoundry_common.fs" -> "typing.Literal"
```

