# `kgforge_common`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [API](#api)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`kgforge_common.config`** — Module for kgforge_common.config → [open](vscode://file//home/paul/KGForge/src/kgforge_common/config.py:1:1) | [view](config.py#L1)
  - **`kgforge_common.config.load_config`** — Load config → [open](vscode://file//home/paul/KGForge/src/kgforge_common/config.py:10:1) | [view](config.py#L10-L20)
- **`kgforge_common.errors`** — Module for kgforge_common.errors → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:1:1) | [view](errors.py#L1)
  - **`kgforge_common.errors.ChunkingError`** — Raised when chunk generation fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:28:1) | [view](errors.py#L28-L31)
  - **`kgforge_common.errors.DoclingError`** — Base Docling processing error → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:16:1) | [view](errors.py#L16-L19)
  - **`kgforge_common.errors.DownloadError`** — Raised when an external download fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:4:1) | [view](errors.py#L4-L7)
  - **`kgforge_common.errors.EmbeddingError`** — Raised when embedding generation fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:34:1) | [view](errors.py#L34-L37)
  - **`kgforge_common.errors.IndexBuildError`** — Raised when index construction fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:46:1) | [view](errors.py#L46-L49)
  - **`kgforge_common.errors.LinkerCalibrationError`** — Raised when linker calibration cannot be performed → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:58:1) | [view](errors.py#L58-L61)
  - **`kgforge_common.errors.Neo4jError`** — Raised when Neo4j operations fail → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:64:1) | [view](errors.py#L64-L67)
  - **`kgforge_common.errors.OCRTimeout`** — Raised when OCR operations exceed time limits → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:22:1) | [view](errors.py#L22-L25)
  - **`kgforge_common.errors.OntologyParseError`** — Raised when ontology parsing fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:52:1) | [view](errors.py#L52-L55)
  - **`kgforge_common.errors.SpladeOOM`** — Raised when SPLADE runs out of memory → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:40:1) | [view](errors.py#L40-L43)
  - **`kgforge_common.errors.UnsupportedMIMEError`** — Raised when an unsupported MIME type is encountered → [open](vscode://file//home/paul/KGForge/src/kgforge_common/errors.py:10:1) | [view](errors.py#L10-L13)
- **`kgforge_common.exceptions`** — Module for kgforge_common.exceptions → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:1:1) | [view](exceptions.py#L1)
  - **`kgforge_common.exceptions.DownloadError`** — Raised when an external download fails → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:4:1) | [view](exceptions.py#L4-L7)
  - **`kgforge_common.exceptions.UnsupportedMIMEError`** — Raised when a MIME type is unsupported → [open](vscode://file//home/paul/KGForge/src/kgforge_common/exceptions.py:10:1) | [view](exceptions.py#L10-L13)
- **`kgforge_common.ids`** — Module for kgforge_common.ids → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:1:1) | [view](ids.py#L1)
  - **`kgforge_common.ids.urn_chunk`** — Urn chunk → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:23:1) | [view](ids.py#L23-L34)
  - **`kgforge_common.ids.urn_doc_from_text`** — Urn doc from text → [open](vscode://file//home/paul/KGForge/src/kgforge_common/ids.py:9:1) | [view](ids.py#L9-L20)
- **`kgforge_common.logging`** — Module for kgforge_common.logging → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:1:1) | [view](logging.py#L1)
  - **`kgforge_common.logging.JsonFormatter`** — Jsonformatter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:8:1) | [view](logging.py#L8-L30)
  - **`kgforge_common.logging.setup_logging`** — Setup logging → [open](vscode://file//home/paul/KGForge/src/kgforge_common/logging.py:33:1) | [view](logging.py#L33-L44)
- **`kgforge_common.models`** — Module for kgforge_common.models → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:1:1) | [view](models.py#L1)
  - **`kgforge_common.models.Chunk`** — Chunk → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:41:1) | [view](models.py#L41-L50)
  - **`kgforge_common.models.Doc`** — Doc → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:12:1) | [view](models.py#L12-L27)
  - **`kgforge_common.models.DoctagsAsset`** — Doctagsasset → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:30:1) | [view](models.py#L30-L38)
  - **`kgforge_common.models.LinkAssertion`** — Linkassertion → [open](vscode://file//home/paul/KGForge/src/kgforge_common/models.py:53:1) | [view](models.py#L53-L63)
- **`kgforge_common.parquet_io`** — Module for kgforge_common.parquet_io → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:1:1) | [view](parquet_io.py#L1)
  - **`kgforge_common.parquet_io.ParquetChunkWriter`** — Parquetchunkwriter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:158:1) | [view](parquet_io.py#L158-L219)
  - **`kgforge_common.parquet_io.ParquetVectorWriter`** — Parquetvectorwriter → [open](vscode://file//home/paul/KGForge/src/kgforge_common/parquet_io.py:17:1) | [view](parquet_io.py#L17-L155)
