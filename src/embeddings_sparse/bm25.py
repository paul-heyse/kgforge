
from __future__ import annotations
import math, os, re, json, pickle
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Iterable, Dict, Tuple, List, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

@dataclass
class BM25Doc:
    doc_id: str
    length: int
    fields: Dict[str, str]

class PurePythonBM25:
    """Simple offline BM25 builder & searcher (Okapi BM25). Persisted as a pickle.
    Fields: title, section, body with configurable boosts."""
    def __init__(self, index_dir: str, k1: float = 0.9, b: float = 0.4, field_boosts: Optional[Dict[str, float]]=None):
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.field_boosts = field_boosts or {"title": 2.0, "section": 1.2, "body": 1.0}
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, Dict[str, int]] = {}
        self.docs: Dict[str, BM25Doc] = {}
        self.N = 0
        self.avgdl = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def build(self, docs_iterable: Iterable[Tuple[str, Dict]]) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        df = defaultdict(int)
        postings = defaultdict(lambda: defaultdict(int))
        docs: Dict[str, BM25Doc] = {}
        lengths = []
        for doc_id, fields in docs_iterable:
            body = fields.get("body","")
            section = fields.get("section","")
            title = fields.get("title","")
            # field boosts applied at scoring time; here we merge for length calc
            toks = self._tokenize(title + " " + section + " " + body)
            lengths.append(len(toks))
            docs[doc_id] = BM25Doc(doc_id=doc_id, length=len(toks), fields={"title":title,"section":section,"body":body})
            seen = set()
            for tok in toks:
                postings[tok][doc_id] += 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)
        self.N = len(docs)
        self.avgdl = sum(lengths)/max(1,len(lengths))
        self.df = dict(df)
        # convert defaultdicts
        self.postings = {t: dict(ps) for t, ps in postings.items()}
        self.docs = docs
        # persist
        with open(os.path.join(self.index_dir, "pure_bm25.pkl"), "wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b, "field_boosts": self.field_boosts,
                "df": self.df, "postings": self.postings,
                "docs": self.docs, "N": self.N, "avgdl": self.avgdl
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        path = os.path.join(self.index_dir, "pure_bm25.pkl")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.k1 = data["k1"]; self.b = data["b"]; self.field_boosts = data["field_boosts"]
        self.df = data["df"]; self.postings = data["postings"]; self.docs = data["docs"]
        self.N = data["N"]; self.avgdl = data["avgdl"]

    def _idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        if n_t == 0:
            return 0.0
        # BM25 idf variant
        return math.log( (self.N - n_t + 0.5) / (n_t + 0.5) + 1.0 )

    def search(self, query: str, k: int, fields: Dict | None = None) -> List[Tuple[str, float]]:
        # naive field weighting at score aggregation (title/section/body contributions)
        tokens = self._tokenize(query)
        scores = defaultdict(float)
        for term in tokens:
            idf = self._idf(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            for doc_id, tf in postings.items():
                doc = self.docs[doc_id]
                dl = doc.length or 1
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                contrib = idf * ( (tf * (self.k1 + 1)) / (denom) )
                scores[doc_id] += contrib
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return ranked

class LuceneBM25:
    """Pyserini-backed Lucene BM25 adapter. Lazily imported."""
    def __init__(self, index_dir: str, k1: float=0.9, b: float=0.4, field_boosts: Optional[Dict[str,float]]=None):
        self.index_dir = index_dir
        self.k1 = k1; self.b = b
        self.field_boosts = field_boosts or {"title":2.0, "section":1.2, "body":1.0}
        self._searcher = None

    def build(self, docs_iterable: Iterable[Tuple[str, Dict]]) -> None:
        try:
            from pyserini.index import IndexWriter, JIndexer
            from pyserini.analysis import get_lucene_analyzer
        except Exception as e:
            raise RuntimeError("Pyserini/Lucene not available") from e
        os.makedirs(self.index_dir, exist_ok=True)
        analyzer = get_lucene_analyzer(stemmer='english', stopwords=True)
        writer = IndexWriter(self.index_dir, analyzer=analyzer, keep_stopwords=False)
        for doc_id, fields in docs_iterable:
            # combine fields with boosts in a "contents" field for simplicity
            title = fields.get("title","")
            section = fields.get("section","")
            body = fields.get("body","")
            contents = " ".join([ (title + " ") * int(self.field_boosts.get("title",1.0)),
                                  (section + " ") * int(self.field_boosts.get("section",1.0)),
                                  body ])
            writer.add_document(docid=doc_id, contents=contents)
        writer.close()

    def _ensure_searcher(self):
        if self._searcher is not None:
            return
        from pyserini.search.lucene import LuceneSearcher
        self._searcher = LuceneSearcher(self.index_dir)
        self._searcher.set_bm25(self.k1, self.b)

    def search(self, query: str, k: int, fields: Dict | None = None) -> List[Tuple[str, float]]:
        self._ensure_searcher()
        hits = self._searcher.search(query, k=k)
        return [(h.docid, float(h.score)) for h in hits]

def get_bm25(backend: str, index_dir: str, **kwargs):
    if backend == "lucene":
        try:
            return LuceneBM25(index_dir, **kwargs)
        except Exception:
            # allow fallback creation
            pass
    return PurePythonBM25(index_dir, **kwargs)
