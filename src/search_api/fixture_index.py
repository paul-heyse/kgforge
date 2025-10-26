
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os, re, math, duckdb
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]

@dataclass
class FixtureDoc:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str

class FixtureIndex:
    def __init__(self, root: str = "/data", db_path: str = "/data/catalog/catalog.duckdb"):
        self.root = Path(root)
        self.db_path = db_path
        self.docs: List[FixtureDoc] = []
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        self._load_from_duckdb()

    def _load_from_duckdb(self) -> None:
        if not Path(self.db_path).exists():
            return
        con = duckdb.connect(self.db_path)
        try:
            ds = con.execute("""
              SELECT parquet_root FROM datasets
              WHERE kind='chunks'
              ORDER BY created_at DESC
              LIMIT 1
            """).fetchone()
            if not ds:
                return
            root = ds[0]
            rows = con.execute(f"""
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text,
                       coalesce(d.title,'') AS title
                FROM read_parquet('{root}/*/*.parquet', union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """).fetchall()
        finally:
            con.close()

        for chunk_id, doc_id, section, text, title in rows:
            self.docs.append(FixtureDoc(chunk_id=chunk_id, doc_id=doc_id or "urn:doc:fixture", title=title or "Fixture", section=section or "", text=text or ""))

        self._build_lex()

    def _build_lex(self) -> None:
        self.tf.clear()
        self.df.clear()
        for doc in self.docs:
            toks = tokenize(doc.text)
            tf_counts: Dict[str, int] = {}
            for t in toks:
                tf_counts[t] = tf_counts.get(t, 0) + 1
            self.tf.append(tf_counts)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        if getattr(self, "N", 0) == 0:
            return []
        qtoks = tokenize(query)
        if not qtoks:
            return []
        scores = [0.0] * self.N
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in qtoks:
                if t not in self.df:
                    continue
                idf = math.log((self.N + 1) / (self.df[t] + 0.5) + 1.0)
                s += idf * tf.get(t, 0)
            scores[i] = s
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0.0]

    def doc(self, idx: int) -> FixtureDoc:
        return self.docs[idx]
