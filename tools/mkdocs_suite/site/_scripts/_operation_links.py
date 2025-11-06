"""Utilities for constructing MkDocs anchors to OpenAPI operations."""

from __future__ import annotations

from urllib.parse import quote

CLI_SPEC_DOC = "../api/openapi-cli.md"
HTTP_SPEC_DOC = "../api/index.md"


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
    if not operation_id or not isinstance(spec_path, str):
        return None
    if spec_path.endswith("openapi-cli.yaml"):
        base_href = CLI_SPEC_DOC
    elif spec_path.endswith("openapi.yaml"):
        base_href = HTTP_SPEC_DOC
    else:
        return None
    encoded_operation = quote(operation_id, safe="")
    return f"{base_href}#operation/{encoded_operation}"
