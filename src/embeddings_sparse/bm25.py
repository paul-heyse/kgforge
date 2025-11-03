"""Overview of bm25.

This module bundles bm25 logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import TYPE_CHECKING, Final, Protocol, cast

if TYPE_CHECKING:
    pass

from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.safe_pickle_v2 import UnsafeSerializationError, load_unsigned_legacy
from kgfoundry_common.serialization import deserialize_json, serialize_json

logger = logging.getLogger(__name__)

__all__ = ["BM25Doc", "LuceneBM25", "PurePythonBM25", "get_bm25"]


def _load_json_metadata(metadata_path: Path, schema_path: Path) -> dict[str, JsonValue]:
    data_raw = deserialize_json(metadata_path, schema_path)
    if not isinstance(data_raw, dict):
        msg = f"Invalid index data format: expected dict, got {type(data_raw)}"
        raise DeserializationError(msg)
    return cast(dict[str, JsonValue], data_raw)


__navmap__: Final[NavMap] = {
    "title": "kgfoundry.embeddings_sparse.bm25",
    "synopsis": "Pure Python and Lucene-backed BM25 adapters for sparse retrieval",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        }
    ],
    "module_meta": {
        "owner": "@embeddings",
        "stability": "experimental",
        "since": "2024.10",
    },
    "symbols": {
        "BM25Doc": {
            "owner": "@embeddings",
            "stability": "experimental",
            "since": "2024.10",
        },
        "PurePythonBM25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
            "tests": [
                "tests/unit/test_bm25_adapter.py::test_bm25_build_and_search_from_fixtures",
            ],
        },
        "LuceneBM25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "experimental",
            "side_effects": ["fs"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
        "get_bm25": {
            "owner": "@embeddings",
            "since": "2024.10",
            "stability": "stable",
            "side_effects": ["none"],
            "thread_safety": "not-threadsafe",
            "async_ok": False,
        },
    },
    "edit_scopes": {"safe": ["get_bm25"], "risky": ["PurePythonBM25", "LuceneBM25"]},
    "tags": ["bm25", "retrieval", "sparse"],
    "see_also": ["kgfoundry.search_api.bm25_index"],
    "deps": ["pyserini"],
}

TOKEN_RE: Pattern[str] = re.compile(r"[A-Za-z0-9_]+")


def _default_int_dict() -> defaultdict[str, int]:
    return defaultdict(int)


class LuceneHitProtocol(Protocol):
    docid: str
    score: float


class LuceneSearcherProtocol(Protocol):
    def set_bm25(self, k1: float, b: float) -> None: ...

    def search(self, query: str, k: int) -> Sequence[LuceneHitProtocol]: ...


class LuceneIndexerProtocol(Protocol):
    def add_doc_dict(self, doc: Mapping[str, str]) -> None: ...

    def close(self) -> None: ...


class LuceneSearcherFactory(Protocol):
    def __call__(self, index_dir: str) -> LuceneSearcherProtocol: ...


class LuceneIndexerFactory(Protocol):
    def __call__(self, index_dir: str) -> LuceneIndexerProtocol: ...


def _score_value(item: tuple[str, float]) -> float:
    return item[1]


# [nav:anchor BM25Doc]
@dataclass
class BM25Doc:
    """Represent a document stored in the in-memory BM25 index.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    doc_id : str
        Describe ``doc_id``.
    length : int
        Describe ``length``.
    fields : dict[str, str]
        Describe ``fields``.
    """

    doc_id: str
    length: int
    fields: dict[str, str]
    term_freqs: dict[str, int] = field(default_factory=dict)


# [nav:anchor PurePythonBM25]
class PurePythonBM25:
    """Pure Python BM25 implementation backed by simple in-memory data structures.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    index_dir : str
        Describe ``index_dir``.
    k1 : float, optional
        Describe ``k1``.
        Defaults to ``0.9``.
    b : float, optional
        Describe ``b``.
        Defaults to ``0.4``.
    field_boosts : dict[str, float] | None, optional
        Describe ``field_boosts``.
        Defaults to ``None``.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Initialise the in-memory BM25 index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        index_dir : str
            Describe ``index_dir``.
        k1 : float, optional
            Describe ``k1``.
            Defaults to ``0.9``.
        b : float, optional
            Describe ``b``.
            Defaults to ``0.4``.
        field_boosts : dict[str, float] | NoneType, optional
            Describe ``field_boosts``.
            Defaults to ``None``.
        """
        self.index_dir = index_dir
        self.k1 = k1
        self.b = b
        self.field_boosts = field_boosts or {"title": 2.0, "section": 1.2, "body": 1.0}
        self.df: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.docs: dict[str, BM25Doc] = {}
        self.N = 0
        self.avgdl = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise text with a simple alphanumeric regex.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        text : str
            Describe ``text``.

        Returns
        -------
        list[str]
            Lowercased tokens extracted from the text.
        """
        matches = cast(list[str], TOKEN_RE.findall(text))
        return [token.lower() for token in matches]

    def build(self, docs_iterable: Iterable[tuple[str, dict[str, str]]]) -> None:
        """Build postings and document statistics for the BM25 index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        docs_iterable : tuple[str, dict[str, str]]
            Describe ``docs_iterable``.
        """
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        df: defaultdict[str, int] = defaultdict(int)
        postings: defaultdict[str, defaultdict[str, int]] = defaultdict(_default_int_dict)
        docs: dict[str, BM25Doc] = {}
        lengths: list[int] = []
        for doc_id, fields in docs_iterable:
            body = fields.get("body", "")
            section = fields.get("section", "")
            title = fields.get("title", "")
            # field boosts applied at scoring time; here we merge for length calc
            toks = self._tokenize(title + " " + section + " " + body)
            lengths.append(len(toks))
            seen = set()
            term_freqs: defaultdict[str, int] = defaultdict(int)
            for tok in toks:
                term_freqs[tok] += 1
                postings[tok][doc_id] += 1
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)
            term_freq_map = {term: int(count) for term, count in term_freqs.items()}
            docs[doc_id] = BM25Doc(
                doc_id=doc_id,
                length=len(toks),
                fields={"title": title, "section": section, "body": body},
                term_freqs=term_freq_map,
            )
        self.N = len(docs)
        self.avgdl = sum(lengths) / max(1, len(lengths))
        self.df = dict(df)
        # convert defaultdicts
        self.postings = {term: dict(term_postings) for term, term_postings in postings.items()}
        self.docs = docs
        # persist using secure JSON serialization with schema validation
        metadata_path = Path(self.index_dir) / "pure_bm25.json"
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "bm25_metadata.v1.json"
        )
        # Convert docs to JSON-serializable format
        docs_data = []
        for doc_id, doc in self.docs.items():
            doc_fields = doc.fields
            doc_record = {
                "chunk_id": doc_id,
                "doc_id": doc_id,
                "title": doc_fields.get("title", ""),
                "section": doc_fields.get("section", ""),
                "body": doc_fields.get("body", ""),
                "tf": doc.term_freqs,
                "dl": float(doc.length),
            }
            docs_data.append(doc_record)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "field_boosts": self.field_boosts,
            "df": self.df,
            "postings": self.postings,
            "docs": docs_data,
            "N": self.N,
            "avgdl": self.avgdl,
        }
        serialize_json(payload, schema_path, metadata_path)

    def load(self) -> None:
        """Load an existing BM25 index from disk with schema validation and checksum verification."""
        payload = self._read_metadata()
        self._initialize_from_payload(payload)

    def _read_metadata(self) -> dict[str, JsonValue]:
        metadata_path = Path(self.index_dir) / "pure_bm25.json"
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "bm25_metadata.v1.json"
        )
        legacy_path = Path(self.index_dir) / "pure_bm25.pkl"

        if metadata_path.exists():
            try:
                return _load_json_metadata(metadata_path, schema_path)
            except DeserializationError as exc:
                logger.warning("Failed to load JSON index, trying legacy pickle: %s", exc)
                if legacy_path.exists():
                    return self._load_legacy_payload(legacy_path)
                raise

        if legacy_path.exists():
            payload = self._load_legacy_payload(legacy_path)
            logger.warning("Loaded legacy pickle index. Consider migrating to JSON format.")
            return payload

        msg = f"Index metadata not found at {metadata_path} or {legacy_path}"
        raise FileNotFoundError(msg)

    @staticmethod
    def _load_legacy_payload(legacy_path: Path) -> dict[str, JsonValue]:
        with legacy_path.open("rb") as handle:
            try:
                payload = load_unsigned_legacy(handle)
            except UnsafeSerializationError as legacy_exc:
                msg = f"Legacy pickle data failed allow-list validation: {legacy_exc}"
                raise DeserializationError(msg) from legacy_exc
        if not isinstance(payload, dict):
            msg = f"Invalid pickle data format: expected dict, got {type(payload)}"
            raise DeserializationError(msg)
        return cast(dict[str, JsonValue], payload)

    def _initialize_from_payload(self, data: Mapping[str, JsonValue]) -> None:
        self._apply_scalar_metadata(data)
        self.docs = self._build_docs_from_metadata(data)
        postings_val = data.get("postings", {})
        self.postings = (
            cast(dict[str, dict[str, int]], postings_val) if isinstance(postings_val, dict) else {}
        )

    def _apply_scalar_metadata(self, data: Mapping[str, JsonValue]) -> None:
        k1_val = data.get("k1", 0.9)
        b_val = data.get("b", 0.4)
        self.k1 = float(k1_val) if isinstance(k1_val, (int, float)) else 0.9
        self.b = float(b_val) if isinstance(b_val, (int, float)) else 0.4
        field_boosts_val = data.get("field_boosts", {"title": 2.0, "section": 1.2, "body": 1.0})
        self.field_boosts = (
            cast(dict[str, float], field_boosts_val)
            if isinstance(field_boosts_val, dict)
            else {"title": 2.0, "section": 1.2, "body": 1.0}
        )
        df_val = data.get("df", {})
        self.df = cast(dict[str, int], df_val) if isinstance(df_val, dict) else {}
        n_val = data.get("N", 0)
        avgdl_val = data.get("avgdl", 0.0)
        self.N = int(n_val) if isinstance(n_val, (int, float)) else 0
        self.avgdl = float(avgdl_val) if isinstance(avgdl_val, (int, float)) else 0.0

    @staticmethod
    def _build_docs_from_metadata(data: Mapping[str, JsonValue]) -> dict[str, BM25Doc]:
        docs_data_raw = data.get("docs", [])
        if isinstance(docs_data_raw, list) and docs_data_raw:
            docs: dict[str, BM25Doc] = {}
            for doc_value in docs_data_raw:
                if not isinstance(doc_value, dict):
                    continue
                doc_id_raw = doc_value.get("doc_id") or doc_value.get("chunk_id")
                doc_id = str(doc_id_raw) if doc_id_raw is not None else ""
                if not doc_id:
                    continue
                length_val = doc_value.get("dl", doc_value.get("length", 0))
                length = int(length_val) if isinstance(length_val, (int, float)) else 0
                title = str(doc_value.get("title", ""))
                section = str(doc_value.get("section", ""))
                body = str(doc_value.get("body", ""))
                tf_raw = doc_value.get("tf", doc_value.get("term_freqs", {}))
                tf_map = (
                    {
                        str(term): int(freq)
                        for term, freq in cast(dict[object, object], tf_raw).items()
                        if isinstance(term, str) and isinstance(freq, (int, float))
                    }
                    if isinstance(tf_raw, dict)
                    else {}
                )
                docs[doc_id] = BM25Doc(
                    doc_id=doc_id,
                    length=length,
                    fields={"title": title, "section": section, "body": body},
                    term_freqs=tf_map,
                )
            return docs

        docs_val = data.get("docs", {})
        if isinstance(docs_val, dict):
            return cast(dict[str, BM25Doc], docs_val)
        return {}

    def _idf(self, term: str) -> float:
        """Compute the inverse document frequency for a given term.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        term : str
            Describe ``term``.

        Returns
        -------
        float
            Inverse document frequency score for the term.
        """
        n_t = self.df.get(term, 0)
        if n_t == 0:
            return 0.0
        # BM25 idf variant
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def search(
        self,
        query: str,
        k: int,
        fields: Mapping[str, str] | None = None,
    ) -> list[tuple[str, float]]:
        """Score documents stored in the in-memory BM25 index.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int
            Describe ``k``.
        fields : str | str | NoneType, optional
            Describe ``fields``.
            Defaults to ``None``.

        Returns
        -------
        list[tuple[str, float]]
            Ranked document identifiers with their BM25 scores.
        """
        # naive field weighting at score aggregation (title/section/body contributions)
        tokens = self._tokenize(query)
        if fields:
            for text in fields.values():
                tokens.extend(self._tokenize(text))
        scores: defaultdict[str, float] = defaultdict(float)
        for term in tokens:
            idf = self._idf(term)
            postings = self.postings.get(term)
            if not postings:
                continue
            for doc_id, tf in postings.items():
                doc = self.docs[doc_id]
                dl = doc.length or 1
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                contrib = idf * ((tf * (self.k1 + 1)) / (denom))
                scores[doc_id] += contrib
        ranked_scores: list[tuple[str, float]] = [
            (doc_id, score) for doc_id, score in scores.items()
        ]
        ranked_scores.sort(key=_score_value, reverse=True)
        return ranked_scores[:k]


# [nav:anchor LuceneBM25]
class LuceneBM25:
    """Wrap Pyserini's Lucene BM25 indexer with project defaults.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    index_dir : str
        Describe ``index_dir``.
    k1 : float, optional
        Describe ``k1``.
        Defaults to ``0.9``.
    b : float, optional
        Describe ``b``.
        Defaults to ``0.4``.
    field_boosts : dict[str, float] | None, optional
        Describe ``field_boosts``.
        Defaults to ``None``.

    Raises
    ------
    RuntimeError
    Raised when Pyserini is not installed in the environment.
    """

    def __init__(
        self,
        index_dir: str,
        k1: float = 0.9,
        b: float = 0.4,
        field_boosts: dict[str, float] | None = None,
    ) -> None:
        """Initialise the Lucene-backed BM25 adapter.

        Parameters
        ----------
        index_dir : str
            Path to the Lucene index directory on disk.
        k1 : float, optional
            BM25 term saturation parameter forwarded to Pyserini. Defaults to ``0.9``.
        b : float, optional
            BM25 document length normalisation parameter. Defaults to ``0.4``.
        field_boosts : dict[str, float] | None, optional
            Optional mapping of field names to boost weights. Defaults to ``None``.

        Raises
        ------
        RuntimeError
            Raised when Pyserini or its Java dependencies are unavailable.
        """
