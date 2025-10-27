"""Module for kgforge_common.exceptions."""


class DownloadError(Exception):
    """Raised when an external download fails."""

    ...


class UnsupportedMIMEError(Exception):
    """Raised when a MIME type is unsupported."""

    ...
