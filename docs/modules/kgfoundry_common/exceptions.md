# kgfoundry_common.exceptions

Legacy exception aliases maintained for backwards compatibility

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/exceptions.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.errors.DownloadError`, `kgfoundry_common.errors.UnsupportedMIMEError`, `kgfoundry_common.navmap_loader.load_nav_metadata`

## Neighborhood

```d2
direction: right
"kgfoundry_common.exceptions": "kgfoundry_common.exceptions" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/exceptions.py" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.exceptions" -> "__future__.annotations"
"kgfoundry_common.errors.DownloadError": "kgfoundry_common.errors.DownloadError"
"kgfoundry_common.exceptions" -> "kgfoundry_common.errors.DownloadError"
"kgfoundry_common.errors.UnsupportedMIMEError": "kgfoundry_common.errors.UnsupportedMIMEError"
"kgfoundry_common.exceptions" -> "kgfoundry_common.errors.UnsupportedMIMEError"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.exceptions" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.exceptions" { style: dashed }
```

