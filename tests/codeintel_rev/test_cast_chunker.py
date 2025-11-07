"""Tests for cAST chunker byte-to-line mapping."""

from pathlib import Path

from codeintel_rev.indexing.cast_chunker import chunk_file
from codeintel_rev.indexing.scip_reader import Range, SymbolDef


def test_chunk_with_mid_line_bounds_uses_correct_line_indices() -> None:
    """Chunk boundaries inside lines should map to the containing lines."""
    text = "alpha = 1\nvalue = compute()\nthird line!\n"
    definitions = [
        SymbolDef(
            symbol="python test#mid_line",
            path="file.py",
            range=Range(
                start_line=1,
                start_character=4,
                end_line=2,
                end_character=5,
            ),
            language="python",
        )
    ]

    chunks = chunk_file(Path("file.py"), text, definitions, budget=2200)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.start_line == 1
    assert chunk.end_line == 2
