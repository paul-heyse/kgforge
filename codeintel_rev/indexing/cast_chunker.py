"""cAST chunking using SCIP symbol ranges.

Implements structure-aware chunking using symbol boundaries from SCIP
rather than tree-sitter parsing. Greedily packs top-level symbols up to
the character budget, splitting large symbols on blank lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel_rev.indexing.scip_reader import Range, SymbolDef


@dataclass(frozen=True)
class Chunk:
    """Code chunk with precise byte and line bounds.

    A Chunk represents a contiguous region of source code that has been extracted
    for indexing. Chunks are created by the cAST chunker using SCIP symbol boundaries
    to ensure that chunks respect semantic structure (functions, classes, etc.) rather
    than arbitrary text splits.

    Each chunk includes precise byte offsets (for exact text extraction) and line
    numbers (for human-readable display). The chunk also tracks which SCIP symbols
    it contains, enabling symbol-aware search and navigation.

    Chunks are immutable (frozen dataclass) to ensure thread safety and prevent
    accidental modification. They are designed to be stored in Parquet files with
    their embeddings for efficient retrieval.

    Attributes
    ----------
    uri : str
        File path or URI identifying the source file. This is typically a relative
        path from the repository root, matching the SCIP index format. Used for
        filtering and grouping chunks by file.
    start_byte : int
        Starting byte offset (0-indexed) of the chunk within the source file.
        Used for precise text extraction without re-parsing the file. Byte offsets
        are more reliable than character offsets for multi-byte encodings.
    end_byte : int
        Ending byte offset (exclusive, 0-indexed) of the chunk. The chunk text
        spans from start_byte to end_byte (exclusive).
    start_line : int
        Starting line number (0-indexed) for human-readable display. Used in search
        results and code navigation. Line numbers are computed from byte offsets
        using line start positions.
    end_line : int
        Ending line number (0-indexed, inclusive). The chunk spans from start_line
        to end_line (inclusive). Used for displaying code ranges in search results.
    text : str
        The actual source code text of the chunk. This is the substring of the
        source file from start_byte to end_byte. Stored explicitly for fast
        retrieval without re-reading files.
    symbols : tuple[str, ...]
        Tuple of SCIP symbol strings that are defined or referenced within this
        chunk. Symbols are in SCIP format (e.g., "python kgfoundry.core#Function.main").
        Used for symbol-aware search and filtering. Empty tuple if no symbols.
    """

    uri: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    text: str
    symbols: tuple[str, ...]


def line_starts(text: str) -> list[int]:
    """Compute byte offsets of each line start.

    Parameters
    ----------
    text : str
        Source file content.

    Returns
    -------
    list[int]
        Byte offsets for start of each line.
    """
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def range_to_bytes(text: str, starts: list[int], rng: Range) -> tuple[int, int]:
    """Convert line/char range to byte offsets.

    Parameters
    ----------
    text : str
        Source file content.
    starts : list[int]
        Line start offsets from line_starts().
    rng : Range
        SCIP range (0-indexed lines and characters).

    Returns
    -------
    tuple[int, int]
        (start_byte, end_byte).
    """
    n = len(text)

    # Start position
    if rng.start_line < len(starts):
        start = min(starts[rng.start_line] + rng.start_character, n)
    else:
        start = n

    # End position
    if rng.end_line < len(starts):
        end = min(starts[rng.end_line] + rng.end_character, n)
    else:
        end = n

    return start, min(end, n)


def chunk_file(
    path: Path,
    text: str,
    definitions: list[SymbolDef],
    budget: int = 2200,
) -> list[Chunk]:
    """Chunk file using SCIP symbol boundaries.

    Greedily packs top-level symbols up to budget; splits large symbols on
    blank lines.

    Parameters
    ----------
    path : Path
        File path.
    text : str
        File content.
    definitions : list[SymbolDef]
        Symbol definitions for this file (should be top-level only).
    budget : int
        Target chunk size in characters.

    Returns
    -------
    list[Chunk]
        Generated chunks.
    """
    if not definitions:
        return []

    uri = str(path)
    starts = line_starts(text)
    n = len(text)

    # Convert ranges to bytes, sort by start
    symbols_with_bytes: list[tuple[SymbolDef, int, int]] = []
    for sym_def in definitions:
        start, end = range_to_bytes(text, starts, sym_def.range)
        symbols_with_bytes.append((sym_def, start, end))

    symbols_with_bytes.sort(key=lambda x: x[1])

    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_syms: list[str] = []

    for sym_def, start, end in symbols_with_bytes:
        piece_len = end - start

        if cur_start is None:
            # Start first chunk
            cur_start, cur_end, cur_syms = start, end, [sym_def.symbol]
            continue

        # Try to merge
        if cur_end is not None and (max(cur_end, end) - cur_start) <= budget:
            cur_end = max(cur_end, end)
            cur_syms.append(sym_def.symbol)
        else:
            # Flush current chunk
            if cur_start is not None and cur_end is not None:
                chunk_text = text[cur_start:cur_end]
                # Find line numbers for chunk
                sl = next(
                    (i for i, ls in enumerate(starts) if ls >= cur_start),
                    len(starts) - 1,
                )
                el = next(
                    (i for i, ls in enumerate(starts) if ls >= cur_end),
                    len(starts) - 1,
                )
                chunks.append(
                    Chunk(
                        uri=uri,
                        start_byte=cur_start,
                        end_byte=cur_end,
                        start_line=sl,
                        end_line=el,
                        text=chunk_text,
                        symbols=tuple(cur_syms),
                    )
                )

            # Handle large symbol
            if piece_len > budget:
                # Split on blank lines
                pos = start
                while pos < end:
                    chunk_end = min(pos + budget, end)
                    # Find nearest blank line boundary
                    search_start = pos + budget // 2
                    search_end = min(chunk_end, end)
                    nl_idx = text.rfind("\n\n", search_start, search_end)
                    if nl_idx != -1 and nl_idx > pos:
                        chunk_end = nl_idx + 1

                    chunk_text = text[pos:chunk_end]
                    sl = next((i for i, ls in enumerate(starts) if ls >= pos), len(starts) - 1)
                    el = next(
                        (i for i, ls in enumerate(starts) if ls >= chunk_end),
                        len(starts) - 1,
                    )
                    chunks.append(
                        Chunk(
                            uri=uri,
                            start_byte=pos,
                            end_byte=chunk_end,
                            start_line=sl,
                            end_line=el,
                            text=chunk_text,
                            symbols=(sym_def.symbol,),
                        )
                    )
                    pos = chunk_end

                cur_start, cur_end, cur_syms = None, None, []
            else:
                cur_start, cur_end, cur_syms = start, end, [sym_def.symbol]

    # Flush remaining
    if cur_start is not None and cur_end is not None:
        chunk_text = text[cur_start:cur_end]
        sl = next((i for i, ls in enumerate(starts) if ls >= cur_start), len(starts) - 1)
        el = next((i for i, ls in enumerate(starts) if ls >= cur_end), len(starts) - 1)
        chunks.append(
            Chunk(
                uri=uri,
                start_byte=cur_start,
                end_byte=cur_end,
                start_line=sl,
                end_line=el,
                text=chunk_text,
                symbols=tuple(cur_syms),
            )
        )

    return chunks


__all__ = [
    "Chunk",
    "chunk_file",
    "line_starts",
    "range_to_bytes",
]
