# orchestration

Typer-powered orchestration command suite covering indexing flows, API bootstrapping,
and end-to-end demonstrations. Each command maps to a generated OpenAPI operation
consumed by the MkDocs suite.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/__init__.py)

## Hierarchy

- **Children:** [orchestration.cli](orchestration/cli.md), [orchestration.cli_context](orchestration/cli_context.md), [orchestration.config](orchestration/config.md), [orchestration.fixture_flow](orchestration/fixture_flow.md), [orchestration.flows](orchestration/flows.md), [orchestration.safe_pickle](orchestration/safe_pickle.md)

## Sections

- **Public API**

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.navmap_types.NavMap`

## Neighborhood

```d2
direction: right
"orchestration": "orchestration" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/__init__.py" }
"__future__.annotations": "__future__.annotations"
"orchestration" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"orchestration" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types.NavMap": "kgfoundry_common.navmap_types.NavMap"
"orchestration" -> "kgfoundry_common.navmap_types.NavMap"
"orchestration.cli": "orchestration.cli" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/cli.py" }
"orchestration" -> "orchestration.cli" { style: dashed }
"orchestration.cli_context": "orchestration.cli_context" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/cli_context.py" }
"orchestration" -> "orchestration.cli_context" { style: dashed }
"orchestration.config": "orchestration.config" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/config.py" }
"orchestration" -> "orchestration.config" { style: dashed }
"orchestration.fixture_flow": "orchestration.fixture_flow" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/fixture_flow.py" }
"orchestration" -> "orchestration.fixture_flow" { style: dashed }
"orchestration.flows": "orchestration.flows" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/flows.py" }
"orchestration" -> "orchestration.flows" { style: dashed }
"orchestration.safe_pickle": "orchestration.safe_pickle" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/safe_pickle.py" }
"orchestration" -> "orchestration.safe_pickle" { style: dashed }
```

