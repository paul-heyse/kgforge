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
from kgfoundry_common.serialization import deserialize_json, serialize_json

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
            dataset = con.execute(
                "SELECT parquet_root FROM datasets "
                "WHERE kind='chunks' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not dataset:
                return index
            root = dataset[0]
            if not isinstance(root, str):
                msg = f"Invalid parquet_root type: {type(root)}"
                raise TypeError(msg)
            # Parameterize query - use pathlib for safe path construction
            root_path = Path(root)
            parquet_pattern = str(root_path / "*" / "*.parquet")
            sql = """
                SELECT c.chunk_id, c.doc_id, coalesce(c.section,''), c.text, coalesce(d.title,'')
                FROM read_parquet(?, union_by_name=true) AS c
                LEFT JOIN documents d ON c.doc_id = d.doc_id
            """
            rows = con.execute(sql, [parquet_pattern]).fetchall()
            # Explicitly type DuckDB query results
            typed_rows: list[tuple[str, str, str, str, str]] = []
            for row in rows:
                chunk_id_val: object = row[0]
                doc_id_val: object = row[1]
                section_val: object = row[2]
                text_val: object = row[3]
                title_val: object = row[4]
                typed_rows.append(
                    (
                        str(chunk_id_val),
                        str(doc_id_val),
                        str(section_val),
                        str(text_val),
                        str(title_val),
                    )
                )
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
            rows = con.execute(sql, [str(resolved_path)]).fetchall()
            # Explicitly type DuckDB query results
            typed_rows: list[tuple[str, str, str, str, str]] = []
            for row in rows:
                chunk_id_val: object = row[0]
                doc_id_val: object = row[1]
                section_val: object = row[2]
                text_val: object = row[3]
                title_val: str = ""  # from_parquet returns empty title
                typed_rows.append(
                    (
                        str(chunk_id_val),
                        str(doc_id_val),
                        str(section_val),
                        str(text_val),
                        title_val,
                    )
                )
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
        try:
            payload_raw = deserialize_json(path_obj, schema_path)
            # Type the payload - deserialize_json returns object (JsonValue at runtime)
            # We validate it's a dict and narrow the type
            if not isinstance(payload_raw, dict):
                payload: dict[str, JsonValue] = {}
            else:
                # Cast to dict[str, JsonValue] since we validated it's a dict
                # JsonValue is object at runtime, so this is safe
                payload = cast(dict[str, JsonValue], payload_raw)
        except DeserializationError:
            # Try legacy pickle format for backward compatibility
            if path_obj.suffix == ".pkl":
                import pickle

                with path_obj.open("rb") as handle:
                    # pickle.load returns object - unavoidable for legacy format support
                    # Use cast to satisfy mypy while maintaining runtime safety via isinstance check
                    legacy_payload_raw: object = cast(object, pickle.load(handle))  # noqa: S301
                    if not isinstance(legacy_payload_raw, dict):
                        payload = {}
                    else:
                        # Cast to JsonValue dict since pickle payload structure matches JSON
                        payload = cast(dict[str, JsonValue], legacy_payload_raw)
            else:
                raise

        # Extract values from payload with proper type narrowing
        # JsonValue types are narrowed at runtime via isinstance checks
        k1_val = payload.get("k1", 0.9)
        b_val = payload.get("b", 0.4)
        index = cls(
            k1=float(k1_val) if isinstance(k1_val, (int, float)) else 0.9,
            b=float(b_val) if isinstance(b_val, (int, float)) else 0.4,
        )
        n_val = payload.get("N", 0)
        avgdl_val = payload.get("avgdl", 0.0)
        df_val = payload.get("df", {})
        index.N = int(n_val) if isinstance(n_val, (int, float)) else 0
        index.avgdl = float(avgdl_val) if isinstance(avgdl_val, (int, float)) else 0.0
        # df_val is dict[str, JsonValue], convert to dict[str, int]
        if isinstance(df_val, dict):
            index.df = {k: int(v) if isinstance(v, (int, float)) else 0 for k, v in df_val.items()}
        else:
            index.df = {}
        # Convert docs data back to BM25Doc objects
        docs_data_raw = payload.get("docs", [])
        if not isinstance(docs_data_raw, list):
            docs_data: list[dict[str, JsonValue]] = []
        else:
            # Narrow to list[dict[str, JsonValue]]
            # isinstance check filters to dict - mypy understands this narrowing
            docs_data = [doc for doc in docs_data_raw if isinstance(doc, dict)]
            # Type annotation for mypy - isinstance in comprehension narrows to dict[str, JsonValue]
            # Cast is needed because list comprehension doesn't preserve type narrowing
            docs_data = cast(list[dict[str, JsonValue]], docs_data)  # type: ignore[redundant-cast]
        # Build BM25Doc objects with type narrowing for each field
        index.docs = [
            BM25Doc(
                chunk_id=(
                    str(doc.get("chunk_id", "")) if isinstance(doc.get("chunk_id"), str) else ""
                ),
                doc_id=str(doc.get("doc_id", "")) if isinstance(doc.get("doc_id"), str) else "",
                title=str(doc.get("title", "")) if isinstance(doc.get("title"), str) else "",
                section=str(doc.get("section", "")) if isinstance(doc.get("section"), str) else "",
                tf=(
                    cast(dict[str, float], doc.get("tf", {}))
                    if isinstance(doc.get("tf"), dict)
                    else {}
                ),
                dl=(
                    float(dl_val) if isinstance(dl_val := doc.get("dl", 0.0), (int, float)) else 0.0
                ),
            )
            for doc in docs_data
        ]
        return index

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
