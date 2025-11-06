# kgfoundry_common.schema_helpers

Schema and model round-trip validation helpers.

## Sections

- **Public API**

## Contents

### kgfoundry_common.schema_helpers.assert_model_roundtrip

::: kgfoundry_common.schema_helpers.assert_model_roundtrip

### kgfoundry_common.schema_helpers.load_schema

::: kgfoundry_common.schema_helpers.load_schema

### kgfoundry_common.schema_helpers.validate_model_against_schema

::: kgfoundry_common.schema_helpers.validate_model_against_schema

## Relationships

**Imports:** `__future__.annotations`, `json`, `kgfoundry_common.errors.DeserializationError`, `kgfoundry_common.errors.SerializationError`, `kgfoundry_common.fs.read_text`, `kgfoundry_common.jsonschema_utils.Draft202012Validator`, `kgfoundry_common.jsonschema_utils.SchemaError`, `kgfoundry_common.jsonschema_utils.ValidationError`, `kgfoundry_common.jsonschema_utils.validate`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.problem_details.JsonValue`, `kgfoundry_common.pydantic.BaseModel`, `pathlib.Path`, `typing.TYPE_CHECKING`, `typing.cast`

## Autorefs Examples

- [kgfoundry_common.schema_helpers.assert_model_roundtrip][]
- [kgfoundry_common.schema_helpers.load_schema][]
- [kgfoundry_common.schema_helpers.validate_model_against_schema][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.schema_helpers": "kgfoundry_common.schema_helpers" { link: "schema_helpers.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.schema_helpers" -> "__future__.annotations"
"json": "json"
"kgfoundry_common.schema_helpers" -> "json"
"kgfoundry_common.errors.DeserializationError": "kgfoundry_common.errors.DeserializationError"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.errors.DeserializationError"
"kgfoundry_common.errors.SerializationError": "kgfoundry_common.errors.SerializationError"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.errors.SerializationError"
"kgfoundry_common.fs.read_text": "kgfoundry_common.fs.read_text"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.fs.read_text"
"kgfoundry_common.jsonschema_utils.Draft202012Validator": "kgfoundry_common.jsonschema_utils.Draft202012Validator"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.jsonschema_utils.Draft202012Validator"
"kgfoundry_common.jsonschema_utils.SchemaError": "kgfoundry_common.jsonschema_utils.SchemaError"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.jsonschema_utils.SchemaError"
"kgfoundry_common.jsonschema_utils.ValidationError": "kgfoundry_common.jsonschema_utils.ValidationError"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.jsonschema_utils.ValidationError"
"kgfoundry_common.jsonschema_utils.validate": "kgfoundry_common.jsonschema_utils.validate"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.jsonschema_utils.validate"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.problem_details.JsonValue": "kgfoundry_common.problem_details.JsonValue"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.problem_details.JsonValue"
"kgfoundry_common.pydantic.BaseModel": "kgfoundry_common.pydantic.BaseModel"
"kgfoundry_common.schema_helpers" -> "kgfoundry_common.pydantic.BaseModel"
"pathlib.Path": "pathlib.Path"
"kgfoundry_common.schema_helpers" -> "pathlib.Path"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"kgfoundry_common.schema_helpers" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"kgfoundry_common.schema_helpers" -> "typing.cast"
```

