
from __future__ import annotations
import hashlib, base64
def urn_doc_from_text(text: str) -> str:
    h = hashlib.sha256(text.encode('utf-8')).digest()[:16]
    b32 = base64.b32encode(h).decode('ascii').strip('=').lower()
    return f"urn:doc:sha256:{b32}"
def urn_chunk(doc_hash: str, start: int, end: int) -> str:
    return f"urn:chunk:{doc_hash.split(':')[-1]}:{start}-{end}"
