from typing import Any, Mapping

NavmapDocument = Mapping[str, Any]
NAVMAP_SCHEMA: str

def navmap_document_from_index(index: Any, *, commit: str, policy_version: str, link_mode: str) -> NavmapDocument: ...
