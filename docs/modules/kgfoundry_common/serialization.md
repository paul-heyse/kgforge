# kgfoundry_common.serialization

Safe serialization helpers with schema validation and checksums.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/serialization.py)

## Hierarchy

- **Parent:** [kgfoundry_common](../kgfoundry_common.md)

## Sections

- **Public API**

## Contents

### kgfoundry_common.serialization._load_data_file

::: kgfoundry_common.serialization._load_data_file

### kgfoundry_common.serialization._load_schema_by_path_str_impl

::: kgfoundry_common.serialization._load_schema_by_path_str_impl

### kgfoundry_common.serialization._load_schema_cached

::: kgfoundry_common.serialization._load_schema_cached

### kgfoundry_common.serialization._validate_json_against_schema

::: kgfoundry_common.serialization._validate_json_against_schema

### kgfoundry_common.serialization._verify_checksum_file

::: kgfoundry_common.serialization._verify_checksum_file

### kgfoundry_common.serialization.compute_checksum

::: kgfoundry_common.serialization.compute_checksum

### kgfoundry_common.serialization.deserialize_json

::: kgfoundry_common.serialization.deserialize_json

### kgfoundry_common.serialization.serialize_json

::: kgfoundry_common.serialization.serialize_json

### kgfoundry_common.serialization.validate_payload

::: kgfoundry_common.serialization.validate_payload

### kgfoundry_common.serialization.verify_checksum

::: kgfoundry_common.serialization.verify_checksum

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Mapping`, `hashlib`, `json`, `kgfoundry_common.errors.DeserializationError`, `kgfoundry_common.errors.SchemaValidationError`, `kgfoundry_common.errors.SerializationError`, `kgfoundry_common.fs.atomic_write`, `kgfoundry_common.fs.read_text`, `kgfoundry_common.fs.write_text`, `kgfoundry_common.jsonschema_utils.SchemaError`, `kgfoundry_common.jsonschema_utils.ValidationError`, `kgfoundry_common.jsonschema_utils.validate`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `pathlib.Path`, `typing.cast`

## Autorefs Examples

- [kgfoundry_common.serialization._load_data_file][]
- [kgfoundry_common.serialization._load_schema_by_path_str_impl][]
- [kgfoundry_common.serialization._load_schema_cached][]

## Neighborhood

```d2
direction: right
"kgfoundry_common.serialization": "kgfoundry_common.serialization" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/serialization.py" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.serialization" -> "__future__.annotations"
"collections.abc.Mapping": "collections.abc.Mapping"
"kgfoundry_common.serialization" -> "collections.abc.Mapping"
"hashlib": "hashlib"
"kgfoundry_common.serialization" -> "hashlib"
"json": "json"
"kgfoundry_common.serialization" -> "json"
"kgfoundry_common.errors.DeserializationError": "kgfoundry_common.errors.DeserializationError"
"kgfoundry_common.serialization" -> "kgfoundry_common.errors.DeserializationError"
"kgfoundry_common.errors.SchemaValidationError": "kgfoundry_common.errors.SchemaValidationError"
"kgfoundry_common.serialization" -> "kgfoundry_common.errors.SchemaValidationError"
"kgfoundry_common.errors.SerializationError": "kgfoundry_common.errors.SerializationError"
"kgfoundry_common.serialization" -> "kgfoundry_common.errors.SerializationError"
"kgfoundry_common.fs.atomic_write": "kgfoundry_common.fs.atomic_write"
"kgfoundry_common.serialization" -> "kgfoundry_common.fs.atomic_write"
"kgfoundry_common.fs.read_text": "kgfoundry_common.fs.read_text"
"kgfoundry_common.serialization" -> "kgfoundry_common.fs.read_text"
"kgfoundry_common.fs.write_text": "kgfoundry_common.fs.write_text"
"kgfoundry_common.serialization" -> "kgfoundry_common.fs.write_text"
"kgfoundry_common.jsonschema_utils.SchemaError": "kgfoundry_common.jsonschema_utils.SchemaError"
"kgfoundry_common.serialization" -> "kgfoundry_common.jsonschema_utils.SchemaError"
"kgfoundry_common.jsonschema_utils.ValidationError": "kgfoundry_common.jsonschema_utils.ValidationError"
"kgfoundry_common.serialization" -> "kgfoundry_common.jsonschema_utils.ValidationError"
"kgfoundry_common.jsonschema_utils.validate": "kgfoundry_common.jsonschema_utils.validate"
"kgfoundry_common.serialization" -> "kgfoundry_common.jsonschema_utils.validate"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.serialization" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.serialization" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"pathlib.Path": "pathlib.Path"
"kgfoundry_common.serialization" -> "pathlib.Path"
"typing.cast": "typing.cast"
"kgfoundry_common.serialization" -> "typing.cast"
"kgfoundry_common": "kgfoundry_common" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/kgfoundry_common/__init__.py" }
"kgfoundry_common" -> "kgfoundry_common.serialization" { style: dashed }
```

