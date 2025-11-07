# download

Downloader command suite that sources external corpora (currently OpenAlex) using the shared
CLI tooling contracts. Emits structured envelopes and metadata so downstream tooling (OpenAPI,
diagrams, documentation) remains in sync without bespoke glue.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/download/__init__.py)

## Hierarchy

- **Children:** [download.cli](download/cli.md), [download.cli_context](download/cli_context.md), [download.harvester](download/harvester.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"download": "download" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/__init__.py" }
"__future__.annotations": "__future__.annotations"
"download" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"download" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"download" -> "kgfoundry_common.navmap_types.NavMap"
"download.cli": "download.cli" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/cli.py" }
"download" -> "download.cli" { style: dashed }
"download.cli_context": "download.cli_context" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/cli_context.py" }
"download" -> "download.cli_context" { style: dashed }
"download.harvester": "download.harvester" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/harvester.py" }
"download" -> "download.harvester" { style: dashed }
```

