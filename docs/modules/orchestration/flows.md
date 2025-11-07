# orchestration.flows

Typer-powered orchestration command suite covering indexing flows, API bootstrapping,
and end-to-end demonstrations. Each command maps to a generated OpenAPI operation
consumed by the MkDocs suite.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/flows.py)

## Hierarchy

- **Parent:** [orchestration](../orchestration.md)

## Sections

- **Public API**

## Contents

### orchestration.flows._e2e_flow_impl

::: orchestration.flows._e2e_flow_impl

### orchestration.flows._t_echo_impl

::: orchestration.flows._t_echo_impl

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `prefect.flow`, `prefect.task`

## Autorefs Examples

- [orchestration.flows._e2e_flow_impl][]
- [orchestration.flows._t_echo_impl][]

## Neighborhood

```d2
direction: right
"orchestration.flows": "orchestration.flows" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/flows.py" }
"__future__.annotations": "__future__.annotations"
"orchestration.flows" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"orchestration.flows" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"prefect.flow": "prefect.flow"
"orchestration.flows" -> "prefect.flow"
"prefect.task": "prefect.task"
"orchestration.flows" -> "prefect.task"
"orchestration": "orchestration" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/orchestration/__init__.py" }
"orchestration" -> "orchestration.flows" { style: dashed }
```

