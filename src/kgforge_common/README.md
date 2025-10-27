# `kgforge_common`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [API](#api)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`kgforge_common.config`** — Module for kgforge_common.config → [open](vscode://file//home/paul/KGForge/src/kgforge_common/config.py:1:1) | [view](config.py#L1)
  - **`kgforge_common.config.load_config`** — Load config → [open](vscode://file//home/paul/KGForge/src/kgforge_common/config.py:14:1) | [view](config.py#L14-L24)
- **`kgforge_common.errors`** — Module for kgforge_common.errors → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:1:1) | [view](errors.py#L1)
  - **`kgforge_common.errors.ChunkingError`** — Raised when chunk generation fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:42:1) | [view](errors.py#L42-L45)
  - **`kgforge_common.errors.DoclingError`** — Base Docling processing error → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:30:1) | [view](errors.py#L30-L33)
  - **`kgforge_common.errors.DownloadError`** — Raised when an external download fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:18:1) | [view](errors.py#L18-L21)
  - **`kgforge_common.errors.EmbeddingError`** — Raised when embedding generation fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:48:1) | [view](errors.py#L48-L51)
  - **`kgforge_common.errors.IndexBuildError`** — Raised when index construction fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:60:1) | [view](errors.py#L60-L63)
  - **`kgforge_common.errors.LinkerCalibrationError`** — Raised when linker calibration cannot be performed → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:72:1) | [view](errors.py#L72-L75)
  - **`kgforge_common.errors.Neo4jError`** — Raised when Neo4j operations fail → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:78:1) | [view](errors.py#L78-L81)
  - **`kgforge_common.errors.OCRTimeout`** — Raised when OCR operations exceed time limits → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:36:1) | [view](errors.py#L36-L39)
  - **`kgforge_common.errors.OntologyParseError`** — Raised when ontology parsing fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:66:1) | [view](errors.py#L66-L69)
  - **`kgforge_common.errors.SpladeOOM`** — Raised when SPLADE runs out of memory → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:54:1) | [view](errors.py#L54-L57)
  - **`kgforge_common.errors.UnsupportedMIMEError`** — Raised when an unsupported MIME type is encountered → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:24:1) | [view](errors.py#L24-L27)
- **`kgforge_common.exceptions`** — Module for kgforge_common.exceptions → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:1:1) | [view](exceptions.py#L1)
  - **`kgforge_common.exceptions.DownloadError`** — Raised when an external download fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:9:1) | [view](exceptions.py#L9-L12)
  - **`kgforge_common.exceptions.UnsupportedMIMEError`** — Raised when a MIME type is unsupported → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:15:1) | [view](exceptions.py#L15-L18)
- **`kgforge_common.ids`** — Module for kgforge_common.ids → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:1:1) | [view](ids.py#L1)
  - **`kgforge_common.ids.urn_chunk`** — Urn chunk → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:28:1) | [view](ids.py#L28-L39)
  - **`kgforge_common.ids.urn_doc_from_text`** — Urn doc from text → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:14:1) | [view](ids.py#L14-L25)
- **`kgforge_common.logging`** — Module for kgforge_common.logging → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:1:1) | [view](logging.py#L1)
  - **`kgforge_common.logging.JsonFormatter`** — Jsonformatter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:13:1) | [view](logging.py#L13-L35)
  - **`kgforge_common.logging.setup_logging`** — Setup logging → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:38:1) | [view](logging.py#L38-L49)
- **`kgforge_common.models`** — Module for kgforge_common.models → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:1:1) | [view](models.py#L1)
  - **`kgforge_common.models.Chunk`** — Chunk → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:48:1) | [view](models.py#L48-L57)
  - **`kgforge_common.models.Doc`** — Doc → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:19:1) | [view](models.py#L19-L34)
  - **`kgforge_common.models.DoctagsAsset`** — Doctagsasset → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:37:1) | [view](models.py#L37-L45)
  - **`kgforge_common.models.LinkAssertion`** — Linkassertion → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:60:1) | [view](models.py#L60-L70)
- **`kgforge_common.parquet_io`** — Module for kgforge_common.parquet_io → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:1:1) | [view](parquet_io.py#L1)
  - **`kgforge_common.parquet_io.ParquetChunkWriter`** — Parquetchunkwriter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:163:1) | [view](parquet_io.py#L163-L224)
  - **`kgforge_common.parquet_io.ParquetVectorWriter`** — Parquetvectorwriter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:22:1) | [view](parquet_io.py#L22-L160)
