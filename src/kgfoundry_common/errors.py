"""Overview of errors.

This module bundles errors logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
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
    """Model the DownloadError.

    Represent the downloaderror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor UnsupportedMIMEError]
class UnsupportedMIMEError(Exception):
    """Model the UnsupportedMIMEError.

    Represent the unsupportedmimeerror data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor DoclingError]
class DoclingError(Exception):
    """Model the DoclingError.

    Represent the doclingerror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor OCRTimeoutError]
class OCRTimeoutError(Exception):
    """Model the OCRTimeoutError.

    Represent the ocrtimeouterror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor ChunkingError]
class ChunkingError(Exception):
    """Model the ChunkingError.

    Represent the chunkingerror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor EmbeddingError]
class EmbeddingError(Exception):
    """Model the EmbeddingError.

    Represent the embeddingerror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor SpladeOOMError]
class SpladeOOMError(Exception):
    """Model the SpladeOOMError.

    Represent the spladeoomerror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor IndexBuildError]
class IndexBuildError(Exception):
    """Model the IndexBuildError.

    Represent the indexbuilderror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor OntologyParseError]
class OntologyParseError(Exception):
    """Model the OntologyParseError.

    Represent the ontologyparseerror data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor LinkerCalibrationError]
class LinkerCalibrationError(Exception):
    """Model the LinkerCalibrationError.

    Represent the linkercalibrationerror data structure used throughout the project. The class
    encapsulates behaviour behind a well-defined interface for collaborating components. Instances
    are typically created by factories or runtime orchestrators documented nearby.
    """

    ...


# [nav:anchor Neo4jError]
class Neo4jError(Exception):
    """Model the Neo4jError.

    Represent the neo4jerror data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    ...
