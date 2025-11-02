"""Overview of bm25 index.

This module bundles bm25 index logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import duckdb

from kgfoundry_common.errors import DeserializationError
from kgfoundry_common.navmap_types import NavMap
from kgfoundry_common.problem_details import JsonValue
from kgfoundry_common.safe_pickle_v2 import UnsafeSerializationError, load_unsigned_legacy
from kgfoundry_common.serialization import deserialize_json, serialize_json
from registry.duckdb_helpers import fetch_all, fetch_one

__all__ = ["BM25Doc", "BM25Index", "toks"]

__navmap__: Final[NavMap] = {
    "title": "search_api.bm25_index",
    "synopsis": "Toy BM25 index backed by DuckDB parquet exports.",
    "exports": __all__,
    "sections": [
        {
            "id": "public-api",
            "title": "Public API",
            "symbols": __all__,
        },
    ],
    "module_meta": {
        "owner": "@search-api",
        "stability": "experimental",
        "since": "0.2.0",
    },
    "symbols": {
        name: {
            "owner": "@search-api",
            "stability": "experimental",
            "since": "0.2.0",
        }
        for name in __all__
    },
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


# [nav:anchor toks]
def toks(text: str) -> list[str]:
    """Describe toks.

    <!-- auto:docstring-builder v1 -->

    Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

    Parameters
    ----------
    text : str
        Describe ``text``.

    Returns
    -------
    list[str]
        Describe return value.
    """
    # re.findall returns list[str] when pattern has no groups
    matches: list[str] = TOKEN_RE.findall(text or "")
    return [token.lower() for token in matches]


# [nav:anchor BM25Doc]
@dataclass
class BM25Doc:
    """Describe BM25Doc.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    chunk_id : str
        Describe ``chunk_id``.
    doc_id : str
        Describe ``doc_id``.
    title : str
        Describe ``title``.
    section : str
        Describe ``section``.
    tf : dict[str, float]
        Describe ``tf``.
    dl : float
        Describe ``dl``.
    """

    chunk_id: str
    doc_id: str
    title: str
    section: str
    tf: dict[str, float]
    dl: float


# [nav:anchor BM25Index]
class BM25Index:
    """Describe BM25Index.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    k1 : float, optional
        Describe ``k1``.
        Defaults to ``0.9``.
    b : float, optional
        Describe ``b``.
        Defaults to ``0.4``.
    """

    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        """Describe   init  .

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        k1 : float, optional
            Describe ``k1``.
            Defaults to ``0.9``.
        b : float, optional
            Describe ``b``.
            Defaults to ``0.4``.
        """
        self.k1 = k1
        self.b = b
        self.docs: list[BM25Doc] = []
        self.df: dict[str, int] = {}
        self.N = 0
        self.avgdl = 0.0

    @classmethod
    def build_from_duckdb(cls, db_path: str) -> BM25Index:
        """Describe build from duckdb.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        db_path : str
            Describe ``db_path``.

        Returns
        -------
        BM25Index
            Describe return value.
        """
        index = cls()
        con = duckdb.connect(db_path)
        try:
            dataset = fetch_one(
                con,
                "SELECT parquet_root FROM datasets "
                "WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1",
            )
            if dataset is None:
                return index
            root_obj = dataset[0]
            if not isinstance(root_obj, str):
                msg = f"Invalid parquet_root type: {type(root_obj)}"
                raise TypeError(msg)
            # Parameterize query - use pathlib for safe path construction
            root_path = Path(root_obj)
            parquet_pattern = str(root_path / "*" / "*.parquet")
            sql = """
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text, coalesce(d.title,'')
                FROM read_parquet(?, union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """
            raw_rows = fetch_all(con, sql, [parquet_pattern])
            typed_rows: list[tuple[str, str, str, str, str]] = [
                (
                    _as_str(chunk_id_val),
                    _as_str(doc_id_val),
                    _as_str(section_val),
                    _as_str(text_val),
                    _as_str(title_val),
                )
                for chunk_id_val, doc_id_val, section_val, text_val, title_val in raw_rows
            ]
        finally:
            con.close()
        index._build(typed_rows)
        return index

    def _build(self, rows: Iterable[tuple[str, str, str, str, str]]) -> None:
        """Describe  build.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        rows : tuple[str, str, str, str, str]
            Describe ``rows``.
        """
        self.docs.clear()
        self.df.clear()
        dl_sum = 0.0
        for chunk_id, doc_id, section, body, title in rows:
            tf: dict[str, float] = {}
            for term in toks(body or ""):
                tf[term] = tf.get(term, 0.0) + 1.0
            for term in toks(title or ""):
                tf[term] = tf.get(term, 0.0) + 2.0
            for term in toks(section or ""):
                tf[term] = tf.get(term, 0.0) + 1.2
            dl = sum(tf.values())
            self.docs.append(
                BM25Doc(
                    chunk_id=chunk_id,
                    doc_id=doc_id or "urn:doc:fixture",
                    title=title or "Fixture",
                    section=section or "",
                    tf=tf,
                    dl=dl,
                )
            )
            dl_sum += dl
            for term in set(tf.keys()):
                self.df[term] = self.df.get(term, 0) + 1
        self.N = len(self.docs)
        self.avgdl = (dl_sum / self.N) if self.N > 0 else 0.0

    @classmethod
    def from_parquet(cls, path: str, *, k1: float = 0.9, b: float = 0.4) -> BM25Index:
        """Describe from parquet.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        path : str
            Describe ``path``.
        k1 : float, optional
            Describe ``k1``.
            Defaults to ``0.9``.
        b : float, optional
            Describe ``b``.
            Defaults to ``0.4``.

        Returns
        -------
        BM25Index
            Describe return value.
        """
        index = cls(k1=k1, b=b)
        con = duckdb.connect(database=":memory:")
        try:
            # Parameterize query - validate and sanitize path input
            path_obj = Path(path)
            resolved_path = path_obj.resolve(strict=True)
            if not resolved_path.exists():
                msg = f"Parquet path not found: {path}"
                raise FileNotFoundError(msg)
            sql = """
                SELECT chunk_id,
                       coalesce(doc_id, chunk_id) AS doc_id,
                       coalesce(section,'') AS section,
                       text,
                       '' AS title
                FROM read_parquet(?, union_by_name=true)
            """
            rows = fetch_all(con, sql, [str(resolved_path)])
            typed_rows: list[tuple[str, str, str, str, str]] = [
                (
                    _as_str(chunk_id_val),
                    _as_str(doc_id_val),
                    _as_str(section_val),
                    _as_str(text_val),
                    "",
                )
                for chunk_id_val, doc_id_val, section_val, text_val, *_ in rows
            ]
        finally:
            con.close()
        index._build(typed_rows)
        return index

    def save(self, path: str) -> None:
        """Save BM25 index metadata to JSON with schema validation and checksum.

        <!-- auto:docstring-builder v1 -->

        Serializes index metadata using secure JSON serialization with schema
        validation and SHA256 checksum for data integrity.

        Parameters
        ----------
        path : str
            Output file path for JSON metadata (will also write .sha256 checksum).

        Raises
        ------
        SerializationError
            If serialization or schema validation fails.
        FileNotFoundError
            If schema file is missing.

        Examples
        --------
        >>> index = BM25Index(k1=0.9, b=0.4)
        >>> index.N = 100
        >>> index.save("/tmp/index.json")
        """
        path_obj = Path(path)
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "bm25_metadata.v1.json"
        )
        # Convert docs to JSON-serializable format
        docs_data = [
            {
                "chunk_id": doc.chunk_id,
                "doc_id": doc.doc_id,
                "title": doc.title,
                "section": doc.section,
                "tf": doc.tf,
                "dl": doc.dl,
            }
            for doc in self.docs
        ]
        payload = {
            "k1": self.k1,
            "b": self.b,
            "N": self.N,
            "avgdl": self.avgdl,
            "df": self.df,
            "docs": docs_data,
        }
        serialize_json(payload, schema_path, path_obj)

    @classmethod
    def load(cls, path: str) -> BM25Index:
        """Load BM25 index metadata from JSON with schema validation and checksum verification.

        <!-- auto:docstring-builder v1 -->

        Deserializes index metadata from JSON, verifying checksum and validating
        against the schema before reconstructing the index.

        Parameters
        ----------
        path : str
            Path to JSON metadata file (checksum file will be verified if present).

        Returns
        -------
        BM25Index
            Reconstructed BM25 index instance.

        Raises
        ------
        DeserializationError
            If deserialization, schema validation, or checksum verification fails.
        FileNotFoundError
            If metadata or schema file is missing.

        Examples
        --------
        >>> index = BM25Index.load("/tmp/index.json")
        >>> assert index.N > 0
        """
        path_obj = Path(path)
        schema_path = (
            Path(__file__).parent.parent.parent / "schema" / "models" / "bm25_metadata.v1.json"
        )
        payload = cls._load_payload(path_obj, schema_path)
        return cls._index_from_payload(payload)

    @staticmethod
    def _coerce_payload(raw: object) -> dict[str, JsonValue]:
        if not isinstance(raw, dict):
            return {}
        return cast(dict[str, JsonValue], raw)

    @classmethod
    def _load_payload(cls, metadata_path: Path, schema_path: Path) -> dict[str, JsonValue]:
        try:
            return cls._coerce_payload(deserialize_json(metadata_path, schema_path))
        except DeserializationError:
            if metadata_path.suffix != ".pkl":
                raise
            return cls._load_legacy_payload(metadata_path)

    @classmethod
    def _load_legacy_payload(cls, metadata_path: Path) -> dict[str, JsonValue]:
        with metadata_path.open("rb") as handle:
            try:
                legacy_payload_raw: object = load_unsigned_legacy(handle)
            except UnsafeSerializationError as legacy_exc:
                message = f"Legacy BM25 pickle failed validation: {legacy_exc}"
                raise DeserializationError(message) from legacy_exc
        return cls._coerce_payload(legacy_payload_raw)

    @classmethod
    def _index_from_payload(cls, payload: dict[str, JsonValue]) -> BM25Index:
        k1_val = payload.get("k1", 0.9)
        b_val = payload.get("b", 0.4)
        index = cls(
            k1=float(k1_val) if isinstance(k1_val, (int, float)) else 0.9,
            b=float(b_val) if isinstance(b_val, (int, float)) else 0.4,
        )
        n_val = payload.get("N", 0)
        avgdl_val = payload.get("avgdl", 0.0)
        index.N = int(n_val) if isinstance(n_val, (int, float)) else 0
        index.avgdl = float(avgdl_val) if isinstance(avgdl_val, (int, float)) else 0.0

        df_val = payload.get("df", {})
        index.df = (
            {k: int(v) if isinstance(v, (int, float)) else 0 for k, v in df_val.items()}
            if isinstance(df_val, dict)
            else {}
        )

        index.docs = cls._docs_from_payload(payload.get("docs", []))
        return index

    @staticmethod
    def _docs_from_payload(raw_docs: object) -> list[BM25Doc]:
        if not isinstance(raw_docs, list):
            return []

        docs: list[BM25Doc] = []
        for entry in raw_docs:
            if not isinstance(entry, dict):
                continue
            chunk_id = entry.get("chunk_id", "")
            doc_id = entry.get("doc_id", "")
            title = entry.get("title", "")
            section = entry.get("section", "")
            tf_value: object = entry.get("tf", {})
            doc_length = entry.get("dl", 0.0)
            docs.append(
                BM25Doc(
                    chunk_id=str(chunk_id) if isinstance(chunk_id, str) else "",
                    doc_id=str(doc_id) if isinstance(doc_id, str) else "",
                    title=str(title) if isinstance(title, str) else "",
                    section=str(section) if isinstance(section, str) else "",
                    tf=(cast(dict[str, float], tf_value) if isinstance(tf_value, dict) else {}),
                    dl=(
                        float(doc_length)
                        if isinstance(doc_length, (int, float))
                        else 0.0
                    ),
                )
            )
        return docs

    def _idf(self, term: str) -> float:
        """Describe  idf.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        term : str
            Describe ``term``.

        Returns
        -------
        float
            Describe return value.
        """
        df = self.df.get(term, 0)
        if self.N == 0 or df == 0:
            return 0.0
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Describe search.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        query : str
            Describe ``query``.
        k : int, optional
            Describe ``k``.
            Defaults to ``10``.

        Returns
        -------
        list[tuple[str, float]]
            Describe return value.
        """
        if self.N == 0:
            return []
        terms = toks(query)
        scores = [0.0] * self.N
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.tf.get(term, 0.0)
                if tf <= 0.0:
                    continue
                idf = self._idf(term)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc.dl / (self.avgdl or 1.0)))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)
            scores[i] = score

        # Explicitly type sorted callable to avoid Any
        def key_func(item: tuple[int, float]) -> float:
            return item[1]

        ranked: list[tuple[int, float]] = sorted(enumerate(scores), key=key_func, reverse=True)
        return [(self.docs[index].chunk_id, score) for index, score in ranked[:k] if score > 0.0]

    def doc(self, index: int) -> BM25Doc:
        """Describe doc.

        <!-- auto:docstring-builder v1 -->

        Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

        Parameters
        ----------
        index : int
            Describe ``index``.

        Returns
        -------
        BM25Doc
            Describe return value.
        """
        return self.docs[index]
