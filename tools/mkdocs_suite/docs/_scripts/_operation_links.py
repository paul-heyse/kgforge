"""Utilities for constructing MkDocs anchors to OpenAPI operations."""

from __future__ import annotations

import os
from urllib.parse import quote

CLI_SPEC_DOC = "api/openapi-cli.md"
HTTP_SPEC_DOC = "api/index.md"


def _normalize_spec_path(spec_path: object) -> str | None:
    """Return a string representation for ``spec_path`` when possible."""

    try:
        resolved = os.fspath(spec_path)
    except TypeError:
        return None
    if not resolved:
        return None
    return str(resolved)


def build_operation_href(spec_path: object, operation_id: str) -> str | None:
    """Return a documentation href for the given OpenAPI operation.

    Parameters
    ----------
    spec_path : object
        Specification file path.
    operation_id : str
        OpenAPI operation identifier.

    Returns
    -------
    str | None
        Fully-qualified anchor URL pointing to the operation in the rendered
        specification, or ``None`` when the specification cannot be resolved.
    """

    if not operation_id:
        return None

    resolved_spec = _normalize_spec_path(spec_path)
    if resolved_spec is None:
        return None

    if resolved_spec.endswith("openapi-cli.yaml"):
        target = CLI_SPEC_DOC
    elif resolved_spec.endswith("openapi.yaml"):
        target = HTTP_SPEC_DOC
    else:
        return None

    encoded_operation = quote(operation_id, safe="")
    return f"{target}#operation/{encoded_operation}"
