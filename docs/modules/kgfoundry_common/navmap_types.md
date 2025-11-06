# kgfoundry_common.navmap_types

Shared navigation metadata structures used across kgfoundry.

## Sections

- **Public API**

## Contents

### kgfoundry_common.navmap_types.ModuleMeta

::: kgfoundry_common.navmap_types.ModuleMeta

*Bases:* TypedDict

### kgfoundry_common.navmap_types.NavMap

::: kgfoundry_common.navmap_types.NavMap

*Bases:* TypedDict

### kgfoundry_common.navmap_types.NavSection

::: kgfoundry_common.navmap_types.NavSection

*Bases:* TypedDict

### kgfoundry_common.navmap_types.SymbolMeta

::: kgfoundry_common.navmap_types.SymbolMeta

*Bases:* TypedDict

## Relationships

**Imports:** `__future__.annotations`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `typing.Literal`, `typing.NotRequired`, `typing.TypedDict`

## Autorefs Examples

- [kgfoundry_common.navmap_types.ModuleMeta][]
- [kgfoundry_common.navmap_types.NavMap][]
- [kgfoundry_common.navmap_types.NavSection][]

## Inheritance

```mermaid
classDiagram
    class ModuleMeta
    class TypedDict
    TypedDict <|-- ModuleMeta
    class NavMap
    TypedDict <|-- NavMap
    class NavSection
    TypedDict <|-- NavSection
    class SymbolMeta
    TypedDict <|-- SymbolMeta
```

## Neighborhood

```d2
direction: right
"kgfoundry_common.navmap_types": "kgfoundry_common.navmap_types" { link: "navmap_types.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.navmap_types" -> "__future__.annotations"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.navmap_types" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"typing.Literal": "typing.Literal"
"kgfoundry_common.navmap_types" -> "typing.Literal"
"typing.NotRequired": "typing.NotRequired"
"kgfoundry_common.navmap_types" -> "typing.NotRequired"
"typing.TypedDict": "typing.TypedDict"
"kgfoundry_common.navmap_types" -> "typing.TypedDict"
```

