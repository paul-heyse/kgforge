# embeddings_dense

Dense embedding adapters and protocols

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_dense/__init__.py)

## Hierarchy

- **Children:** [embeddings_dense.base](embeddings_dense/base.md), [embeddings_dense.qwen3](embeddings_dense/qwen3.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"embeddings_dense": "embeddings_dense" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_dense/__init__.py" }
"__future__.annotations": "__future__.annotations"
"embeddings_dense" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"embeddings_dense" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"embeddings_dense" -> "kgfoundry_common.navmap_types.NavMap"
"embeddings_dense.base": "embeddings_dense.base" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_dense/base.py" }
"embeddings_dense" -> "embeddings_dense.base" { style: dashed }
"embeddings_dense.qwen3": "embeddings_dense.qwen3" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_dense/qwen3.py" }
"embeddings_dense" -> "embeddings_dense.qwen3" { style: dashed }
```

