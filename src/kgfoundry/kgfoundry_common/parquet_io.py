"""Expose ``kgfoundry_common.parquet_io`` inside the ``kgfoundry``
namespace.
"""

from __future__ import annotations

from kgfoundry_common.parquet_io import ParquetChunkWriter, ParquetVectorWriter

__all__ = ["ParquetChunkWriter", "ParquetVectorWriter"]
