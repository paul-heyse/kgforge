# kgfoundry_common.ids

Helpers for generating deterministic URNs

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/ids.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)

## Sections

- **Public API**

## Contents

### kgfoundry_common.ids.urn_chunk

::: kgfoundry_common.ids.urn_chunk

### kgfoundry_common.ids.urn_doc_from_text

::: kgfoundry_common.ids.urn_doc_from_text

## Relationships

**Imports:** `__future__.annotations`, `base64`, `hashlib`, `kgfoundry_common.navmap_loader.load_nav_metadata`

## Autorefs Examples

- [kgfoundry_common.ids.urn_chunk][]
- [kgfoundry_common.ids.urn_doc_from_text][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.ids": "kgfoundry_common.ids" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/ids.py" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.ids" -> "__future__.annotations"
"base64": "base64"
"kgfoundry_common.ids" -> "base64"
"hashlib": "hashlib"
"kgfoundry_common.ids" -> "hashlib"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.ids" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.ids" { style: dashed }
```

