# vectorstore_faiss

FAISS GPU vector store wrappers

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/vectorstore_faiss/__init__.py)

## Hierarchy

- **Children:** [vectorstore_faiss.gpu](vectorstore_faiss/gpu.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"vectorstore_faiss": "vectorstore_faiss" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/vectorstore_faiss/__init__.py" }
"__future__.annotations": "__future__.annotations"
"vectorstore_faiss" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"vectorstore_faiss" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"vectorstore_faiss" -> "kgfoundry_common.navmap_types.NavMap"
"vectorstore_faiss.gpu": "vectorstore_faiss.gpu" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/vectorstore_faiss/gpu.py" }
"vectorstore_faiss" -> "vectorstore_faiss.gpu" { style: dashed }
```

