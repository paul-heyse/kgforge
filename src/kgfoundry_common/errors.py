"""Overview of errors.

This module bundles errors logic for the kgfoundry stack. It groups
related helpers so downstream packages can import a single cohesive
namespace. Refer to the functions and classes below for implementation
specifics.
"""

from __future__ import annotations

from typing import Final

from kgfoundry_common.navmap_types import NavMap

__all__ = [
    "ChunkingError",
    "DoclingError",
    "DownloadError",
    "EmbeddingError",
    "IndexBuildError",
    "LinkerCalibrationError",
    "Neo4jError",
    "OCRTimeoutError",
    "OntologyParseError",
    "SpladeOOMError",
    "UnsupportedMIMEError",
]

__navmap__: Final[NavMap] = {
    "title": "kgfoundry_common.errors",
    "synopsis": "Common error types shared across kgfoundry",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@kgfoundry-common",
        "stability": "stable",
        "since": "0.1.0",
    },
    "symbols": {
        name: {
            "owner": "@kgfoundry-common",
            "stability": "stable",
            "since": "0.1.0",
        }
        for name in __all__
    },
}


# [nav:anchor DownloadError]
class DownloadError(Exception):
    """Describe DownloadError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Describe UnsupportedMIMEError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor DoclingError]
class DoclingError(Exception):
    """Describe DoclingError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor OCRTimeoutError]
class OCRTimeoutError(Exception):
    """Describe OCRTimeoutError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor ChunkingError]
class ChunkingError(Exception):
    """Describe ChunkingError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor EmbeddingError]
class EmbeddingError(Exception):
    """Describe EmbeddingError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor SpladeOOMError]
class SpladeOOMError(Exception):
    """Describe SpladeOOMError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor IndexBuildError]
class IndexBuildError(Exception):
    """Describe IndexBuildError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor OntologyParseError]
class OntologyParseError(Exception):
    """Describe OntologyParseError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor LinkerCalibrationError]
class LinkerCalibrationError(Exception):
    """Describe LinkerCalibrationError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...


# [nav:anchor Neo4jError]
class Neo4jError(Exception):
    """Describe Neo4jError.

<!-- auto:docstring-builder v1 -->
Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
"""

    ...
