# linking

Entity linking calibration and production pipelines

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/linking/__init__.py)

## Hierarchy

- **Children:** [linking.calibration](linking/calibration.md), [linking.linker](linking/linker.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"linking": "linking" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/linking/__init__.py" }
"__future__.annotations": "__future__.annotations"
"linking" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"linking" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"linking" -> "kgfoundry_common.navmap_types.NavMap"
"linking.calibration": "linking.calibration" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/linking/calibration.py" }
"linking" -> "linking.calibration" { style: dashed }
"linking.linker": "linking.linker" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/linking/linker.py" }
"linking" -> "linking.linker" { style: dashed }
```

