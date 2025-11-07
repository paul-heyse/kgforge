# kgfoundry_common.fs

Shared utilities and data structures used across KgFoundry services and tools.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/fs.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)

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
"kgfoundry_common.fs": "kgfoundry_common.fs" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/fs.py" }
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
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.fs" { style: dashed }
```

