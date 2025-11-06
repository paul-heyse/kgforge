# kgfoundry_common.sequence_guards

Sequence access guards for observability-critical code paths.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/sequence_guards.py)

## Sections

- **Public API**

## Contents

### kgfoundry_common.sequence_guards.first_or_error

::: kgfoundry_common.sequence_guards.first_or_error

### kgfoundry_common.sequence_guards.first_or_error_multi_device

::: kgfoundry_common.sequence_guards.first_or_error_multi_device

## Relationships

**Imports:** `__future__.annotations`, `collections.abc`, `collections.abc.Sequence`, `kgfoundry_common.errors.VectorSearchError`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.logging.with_fields`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.problem_details.build_problem_details`, `kgfoundry_common.problem_details.render_problem`, `typing.TYPE_CHECKING`

## Autorefs Examples

- [kgfoundry_common.sequence_guards.first_or_error][]
- [kgfoundry_common.sequence_guards.first_or_error_multi_device][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.sequence_guards": "kgfoundry_common.sequence_guards" { link: "./kgfoundry_common/sequence_guards.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.sequence_guards" -> "__future__.annotations"
"collections.abc": "collections.abc"
"kgfoundry_common.sequence_guards" -> "collections.abc"
"collections.abc.Sequence": "collections.abc.Sequence"
"kgfoundry_common.sequence_guards" -> "collections.abc.Sequence"
"kgfoundry_common.errors.VectorSearchError": "kgfoundry_common.errors.VectorSearchError"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.errors.VectorSearchError"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.logging.with_fields": "kgfoundry_common.logging.with_fields"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.logging.with_fields"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.problem_details.build_problem_details": "kgfoundry_common.problem_details.build_problem_details"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.problem_details.build_problem_details"
"kgfoundry_common.problem_details.render_problem": "kgfoundry_common.problem_details.render_problem"
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.problem_details.render_problem"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"kgfoundry_common.sequence_guards" -> "typing.TYPE_CHECKING"
"kgfoundry_common.sequence_guards_code": "kgfoundry_common.sequence_guards code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/sequence_guards.py" }
"kgfoundry_common.sequence_guards" -> "kgfoundry_common.sequence_guards_code" { style: dashed }
```

