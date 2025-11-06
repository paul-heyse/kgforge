# kgfoundry_common.errors.exceptions

Typed exception hierarchy with Problem Details support.

[View source on GitHub](https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/errors/exceptions.py)

## Sections

- **Public API**

## Contents

### kgfoundry_common.errors.exceptions.AgentCatalogSearchError

::: kgfoundry_common.errors.exceptions.AgentCatalogSearchError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ArtifactDependencyError

::: kgfoundry_common.errors.exceptions.ArtifactDependencyError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ArtifactDeserializationError

::: kgfoundry_common.errors.exceptions.ArtifactDeserializationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ArtifactModelError

::: kgfoundry_common.errors.exceptions.ArtifactModelError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ArtifactSerializationError

::: kgfoundry_common.errors.exceptions.ArtifactSerializationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ArtifactValidationError

::: kgfoundry_common.errors.exceptions.ArtifactValidationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.CatalogLoadError

::: kgfoundry_common.errors.exceptions.CatalogLoadError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.CatalogSessionError

::: kgfoundry_common.errors.exceptions.CatalogSessionError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ChunkingError

::: kgfoundry_common.errors.exceptions.ChunkingError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.ConfigurationError

::: kgfoundry_common.errors.exceptions.ConfigurationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.DeserializationError

::: kgfoundry_common.errors.exceptions.DeserializationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.DoclingError

::: kgfoundry_common.errors.exceptions.DoclingError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.DownloadError

::: kgfoundry_common.errors.exceptions.DownloadError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.EmbeddingError

::: kgfoundry_common.errors.exceptions.EmbeddingError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.IndexBuildError

::: kgfoundry_common.errors.exceptions.IndexBuildError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.KgFoundryError

::: kgfoundry_common.errors.exceptions.KgFoundryError

*Bases:* Exception

### kgfoundry_common.errors.exceptions.KgFoundryErrorConfig

::: kgfoundry_common.errors.exceptions.KgFoundryErrorConfig

### kgfoundry_common.errors.exceptions.LinkerCalibrationError

::: kgfoundry_common.errors.exceptions.LinkerCalibrationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.Neo4jError

::: kgfoundry_common.errors.exceptions.Neo4jError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.OCRTimeoutError

::: kgfoundry_common.errors.exceptions.OCRTimeoutError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.OntologyParseError

::: kgfoundry_common.errors.exceptions.OntologyParseError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.RegistryError

::: kgfoundry_common.errors.exceptions.RegistryError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.RetryExhaustedError

::: kgfoundry_common.errors.exceptions.RetryExhaustedError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.SchemaValidationError

::: kgfoundry_common.errors.exceptions.SchemaValidationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.SerializationError

::: kgfoundry_common.errors.exceptions.SerializationError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.SettingsError

::: kgfoundry_common.errors.exceptions.SettingsError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.SpladeOOMError

::: kgfoundry_common.errors.exceptions.SpladeOOMError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.SymbolAttachmentError

::: kgfoundry_common.errors.exceptions.SymbolAttachmentError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.UnsupportedMIMEError

::: kgfoundry_common.errors.exceptions.UnsupportedMIMEError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions.VectorSearchError

::: kgfoundry_common.errors.exceptions.VectorSearchError

*Bases:* KgFoundryError

### kgfoundry_common.errors.exceptions._coerce_error_config

::: kgfoundry_common.errors.exceptions._coerce_error_config

## Relationships

**Imports:** `__future__.annotations`, `collections.abc.Mapping`, `dataclasses.dataclass`, `kgfoundry_common.errors.codes.ErrorCode`, `kgfoundry_common.errors.codes.get_type_uri`, `kgfoundry_common.logging.get_logger`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `kgfoundry_common.problem_details.JsonValue`, `kgfoundry_common.problem_details.ProblemDetails`, `kgfoundry_common.problem_details.build_problem_details`, `logging`, `typing.TYPE_CHECKING`, `typing.cast`

## Autorefs Examples

- [kgfoundry_common.errors.exceptions.AgentCatalogSearchError][]
- [kgfoundry_common.errors.exceptions.ArtifactDependencyError][]
- [kgfoundry_common.errors.exceptions.ArtifactDeserializationError][]
- [kgfoundry_common.errors.exceptions._coerce_error_config][]

## Inheritance

```mermaid
classDiagram
    class AgentCatalogSearchError
    class KgFoundryError
    KgFoundryError <|-- AgentCatalogSearchError
    class ArtifactDependencyError
    KgFoundryError <|-- ArtifactDependencyError
    class ArtifactDeserializationError
    KgFoundryError <|-- ArtifactDeserializationError
    class ArtifactModelError
    KgFoundryError <|-- ArtifactModelError
    class ArtifactSerializationError
    KgFoundryError <|-- ArtifactSerializationError
    class ArtifactValidationError
    KgFoundryError <|-- ArtifactValidationError
    class CatalogLoadError
    KgFoundryError <|-- CatalogLoadError
    class CatalogSessionError
    KgFoundryError <|-- CatalogSessionError
    class ChunkingError
    KgFoundryError <|-- ChunkingError
    class ConfigurationError
    KgFoundryError <|-- ConfigurationError
    class DeserializationError
    KgFoundryError <|-- DeserializationError
    class DoclingError
    KgFoundryError <|-- DoclingError
    class DownloadError
    KgFoundryError <|-- DownloadError
    class EmbeddingError
    KgFoundryError <|-- EmbeddingError
    class IndexBuildError
    KgFoundryError <|-- IndexBuildError
    class KgFoundryError_1
    class Exception
    Exception <|-- KgFoundryError_1
    class KgFoundryErrorConfig
    class LinkerCalibrationError
    KgFoundryError <|-- LinkerCalibrationError
    class Neo4jError
    KgFoundryError <|-- Neo4jError
    class OCRTimeoutError
    KgFoundryError <|-- OCRTimeoutError
    class OntologyParseError
    KgFoundryError <|-- OntologyParseError
    class RegistryError
    KgFoundryError <|-- RegistryError
    class RetryExhaustedError
    KgFoundryError <|-- RetryExhaustedError
    class SchemaValidationError
    KgFoundryError <|-- SchemaValidationError
    class SerializationError
    KgFoundryError <|-- SerializationError
    class SettingsError
    KgFoundryError <|-- SettingsError
    class SpladeOOMError
    KgFoundryError <|-- SpladeOOMError
    class SymbolAttachmentError
    KgFoundryError <|-- SymbolAttachmentError
    class UnsupportedMIMEError
    KgFoundryError <|-- UnsupportedMIMEError
    class VectorSearchError
    KgFoundryError <|-- VectorSearchError
```

## Neighborhood

```d2
direction: right
"kgfoundry_common.errors.exceptions": "kgfoundry_common.errors.exceptions" { link: "./kgfoundry_common/errors/exceptions.md" }
"__future__.annotations": "__future__.annotations"
"kgfoundry_common.errors.exceptions" -> "__future__.annotations"
"collections.abc.Mapping": "collections.abc.Mapping"
"kgfoundry_common.errors.exceptions" -> "collections.abc.Mapping"
"dataclasses.dataclass": "dataclasses.dataclass"
"kgfoundry_common.errors.exceptions" -> "dataclasses.dataclass"
"kgfoundry_common.errors.codes.ErrorCode": "kgfoundry_common.errors.codes.ErrorCode"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.errors.codes.ErrorCode"
"kgfoundry_common.errors.codes.get_type_uri": "kgfoundry_common.errors.codes.get_type_uri"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.errors.codes.get_type_uri"
"kgfoundry_common.logging.get_logger": "kgfoundry_common.logging.get_logger"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.logging.get_logger"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"kgfoundry_common.problem_details.JsonValue": "kgfoundry_common.problem_details.JsonValue"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.problem_details.JsonValue"
"kgfoundry_common.problem_details.ProblemDetails": "kgfoundry_common.problem_details.ProblemDetails"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.problem_details.ProblemDetails"
"kgfoundry_common.problem_details.build_problem_details": "kgfoundry_common.problem_details.build_problem_details"
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.problem_details.build_problem_details"
"logging": "logging"
"kgfoundry_common.errors.exceptions" -> "logging"
"typing.TYPE_CHECKING": "typing.TYPE_CHECKING"
"kgfoundry_common.errors.exceptions" -> "typing.TYPE_CHECKING"
"typing.cast": "typing.cast"
"kgfoundry_common.errors.exceptions" -> "typing.cast"
"kgfoundry_common.errors.exceptions_code": "kgfoundry_common.errors.exceptions code" { link: "https://github.com/kgfoundry/kgfoundry/blob/main/src/kgfoundry_common/errors/exceptions.py" }
"kgfoundry_common.errors.exceptions" -> "kgfoundry_common.errors.exceptions_code" { style: dashed }
```

