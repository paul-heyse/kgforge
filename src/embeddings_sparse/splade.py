"""Module for embeddings_sparse.splade."""


from __future__ import annotations
import os, re, math, pickle
from typing import List, Tuple, Dict, Iterable, Optional
from collections import defaultdict, Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

class SPLADEv3Encoder:
    """Spladev3encoder."""
    name = "SPLADE-v3-distilbert"
    def __init__(self, model_id: str="naver/splade-v3-distilbert", device: str="cuda", topk: int=256, max_seq_len: int=512):
        """Init.

        Args:
            model_id (str): TODO.
            device (str): TODO.
            topk (int): TODO.
            max_seq_len (int): TODO.
        """
        self.model_id = model_id
        self.device = device
        self.topk = topk
        self.max_seq_len = max_seq_len
    def encode(self, texts: List[str]) -> List[Tuple[list[int], list[float]]]:
        """Encode.

        Args:
            texts (List[str]): TODO.

        Returns:
            List[Tuple[list[int], list[float]]]: TODO.
        """
        # Placeholder in this skeleton. Real impl would run the HF model and produce top-k (token_id, weight).
        raise NotImplementedError("SPLADE encoding is not implemented in the skeleton. Use Lucene impact index if available.")

class PureImpactIndex:
    """Toy 'impact' index that approximates SPLADE using IDF*log(1+tf) weights
    over a simple tokenizer.

    This is *not* a neural encoder; it exists to keep the pipeline
    runnable without GPUs or Pyserini.
    """
    def __init__(self, index_dir: str):
        """Init.

        Args:
            index_dir (str): TODO.
        """
        self.index_dir = index_dir
        self.df: Dict[str,int] = {}
        self.N = 0
        self.postings: Dict[str, Dict[str, float]] = {}
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize.

        Args:
            text (str): TODO.

        Returns:
            List[str]: TODO.
        """
        return [t.lower() for t in TOKEN_RE.findall(text)]
    def build(self, docs_iterable: Iterable[Tuple[str, Dict]]) -> None:
        """Build.

        Args:
            docs_iterable (Iterable[Tuple[str, Dict]]): TODO.

        Returns:
            None: TODO.
        """
        os.makedirs(self.index_dir, exist_ok=True)
        df = defaultdict(int)
        postings = defaultdict(lambda: defaultdict(float))
        N = 0
        for doc_id, fields in docs_iterable:
            text = " ".join([fields.get("title",""), fields.get("section",""), fields.get("body","")])
            toks = self._tokenize(text)
            N += 1
            c = Counter(toks)
            for tok, tf in c.items():
                df[tok] += 1
                postings[tok][doc_id] = math.log1p(tf)
        # compute idf and impact weights
        self.N = N
        self.df = dict(df)
        self.postings = {tok: {doc: w * math.log((N - df[tok] + 0.5)/(df[tok]+0.5) + 1.0)
                               for doc, w in docs.items()} for tok, docs in postings.items()}
        with open(os.path.join(self.index_dir, "impact.pkl"), "wb") as f:
            pickle.dump({"df": self.df, "N": self.N, "postings": self.postings}, f, protocol=pickle.HIGHEST_PROTOCOL)
    def load(self) -> None:
        """Load.

        Returns:
            None: TODO.
        """
        with open(os.path.join(self.index_dir, "impact.pkl"), "rb") as f:
            data = pickle.load(f)
        self.df = data["df"]; self.N = data["N"]; self.postings = data["postings"]
    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Search.

        Args:
            query (str): TODO.
            k (int): TODO.

        Returns:
            List[Tuple[str, float]]: TODO.
        """
        toks = self._tokenize(query)
        scores = defaultdict(float)
        for t in toks:
            posts = self.postings.get(t); 
            if not posts: continue
            for doc, w in posts.items():
                scores[doc] += w
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

class LuceneImpactIndex:
    """Pyserini SPLADE impact index wrapper.

    Requires Pyserini build step that writes an impact index on disk.
    """
    def __init__(self, index_dir: str):
        """Init.

        Args:
            index_dir (str): TODO.
        """
        self.index_dir = index_dir
        self._searcher = None
    def _ensure(self):
        """Ensure."""
        if self._searcher is not None: return
        try:
            from pyserini.search.lucene import LuceneImpactSearcher
        except Exception as e:
            raise RuntimeError("Pyserini not available for SPLADE impact search") from e
        self._searcher = LuceneImpactSearcher(self.index_dir)
    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Search.

        Args:
            query (str): TODO.
            k (int): TODO.

        Returns:
            List[Tuple[str, float]]: TODO.
        """
        self._ensure()
        hits = self._searcher.search(query, k=k)  # expects query to be SPLADE-encoded string
        return [(h.docid, float(h.score)) for h in hits]

def get_splade(backend: str, index_dir: str):
    """Get splade.

    Args:
        backend (str): TODO.
        index_dir (str): TODO.
    """
    if backend == "lucene":
        try:
            return LuceneImpactIndex(index_dir)
        except Exception:
            pass
    return PureImpactIndex(index_dir)
