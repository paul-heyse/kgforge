from collections.abc import Mapping
from typing import Any

NavmapDocument = Mapping[str, Any]
NAVMAP_SCHEMA: str

def navmap_document_from_index(
    index: Any, *, commit: str, policy_version: str, link_mode: str
) -> NavmapDocument: ...
