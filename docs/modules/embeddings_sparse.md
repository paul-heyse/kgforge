# embeddings_sparse

Sparse embedding adapters and indices

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_sparse/__init__.py)

## Hierarchy

- **Children:** [embeddings_sparse.base](embeddings_sparse/base.md), [embeddings_sparse.bm25](embeddings_sparse/bm25.md), [embeddings_sparse.splade](embeddings_sparse/splade.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"embeddings_sparse": "embeddings_sparse" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_sparse/__init__.py" }
"__future__.annotations": "__future__.annotations"
"embeddings_sparse" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"embeddings_sparse" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"embeddings_sparse" -> "kgfoundry_common.navmap_types.NavMap"
"embeddings_sparse.base": "embeddings_sparse.base" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_sparse/base.py" }
"embeddings_sparse" -> "embeddings_sparse.base" { style: dashed }
"embeddings_sparse.bm25": "embeddings_sparse.bm25" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_sparse/bm25.py" }
"embeddings_sparse" -> "embeddings_sparse.bm25" { style: dashed }
"embeddings_sparse.splade": "embeddings_sparse.splade" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/embeddings_sparse/splade.py" }
"embeddings_sparse" -> "embeddings_sparse.splade" { style: dashed }
```

