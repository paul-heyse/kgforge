
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re, math, duckdb
from pathlib import Path
TOKEN = re.compile(r"[A-Za-z0-9]+")
def tok(s: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(s or "")]

@dataclass
class SpladeDoc:
    chunk_id: str
    doc_id: str
    section: str
    text: str

class SpladeIndex:
    def __init__(self, db_path: str, chunks_dataset_root: Optional[str] = None, sparse_root: Optional[str] = None):
        self.db_path = db_path
        self.docs: List[SpladeDoc] = []
        self.df: Dict[str, int] = {}
        self.N = 0
        self._load(chunks_dataset_root)

    def _load(self, chunks_root: Optional[str]):
        if not Path(self.db_path).exists():
            return
        con = duckdb.connect(self.db_path)
        try:
            ds = con.execute("""
                SELECT parquet_root FROM datasets WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1
            """).fetchone()
            if ds:
                rows = con.execute(f"""
                    SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text
                    FROM read_parquet('{ds[0]}/*/*.parquet', union_by_name=true) AS c
                """).fetchall()
                for r in rows:
                    self.docs.append(SpladeDoc(chunk_id=r[0], doc_id=r[1] or "urn:doc:fixture", section=r[2], text=r[3]))
        finally:
            con.close()
        self.N = len(self.docs)
        for d in self.docs:
            for t in set(tok(d.text)):
                self.df[t] = self.df.get(t,0)+1

    def search(self, query: str, k: int=10) -> List[Tuple[int, float]]:
        if self.N == 0: return []
        q = tok(query)
        if not q: return []
        scores = [0.0]*self.N
        for i, d in enumerate(self.docs):
            tf = {}
            for t in tok(d.text): tf[t] = tf.get(t,0)+1
            s = 0.0
            for t in q:
                if t in self.df:
                    idf = (self.N+1)/(self.df[t]+0.5)
                    s += (tf.get(t,0)) * idf
            scores[i]=s
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i,s) for i,s in ranked[:k] if s>0.0]

    def doc(self, i: int) -> SpladeDoc:
        return self.docs[i]
