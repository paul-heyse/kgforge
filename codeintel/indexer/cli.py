"""Command-line utilities for the Tree-sitter-based code-intel indexer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from codeintel.indexer.tscore import LANGUAGE_ALIAS, load_langs, parse_bytes, run_query

app = typer.Typer(no_args_is_help=True)

DEFAULT_NAMED_ONLY: Final[bool] = True

LanguageOption = Annotated[
    str,
    typer.Option(
        "python",
        "--language",
        "-l",
        help="Tree-sitter language to use for parsing the input file.",
        show_default=True,
    ),
]
QueryFileOption = Annotated[
    Path,
    typer.Option(
        ...,
        "--query",
        "-q",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to a Tree-sitter S-expression query file.",
    ),
]
NamedOnlyOption = Annotated[
    bool,
    typer.Option(
        DEFAULT_NAMED_ONLY,
        "--named-only/--all-captures",
        help="Return only named captures unless --all-captures is supplied.",
        show_default=True,
    ),
]


@app.command("query")
def query(
    path: Path, language: LanguageOption, query_file: QueryFileOption, named_only: NamedOnlyOption
) -> None:
    """Execute a Tree-sitter query against a source file.

    Parameters
    ----------
    path : Path
        Source file to parse.
    language : str
        Identifier of the Tree-sitter grammar to use.
    query_file : Path
        Path to a query file containing S-expression patterns.
    named_only : bool
        When ``True`` only named captures are included in the results.

    Raises
    ------
    typer.BadParameter
        If an unsupported language is requested.
    """
    langs = load_langs()
    try:
        lang_attr = LANGUAGE_ALIAS[language]
    except KeyError as exc:
        supported = ", ".join(sorted(LANGUAGE_ALIAS))
        message = f"Unsupported language '{language}'. Supported values: {supported}"
        raise typer.BadParameter(message) from exc
    lang = getattr(langs, lang_attr)
    data = path.read_bytes()
    tree = parse_bytes(lang, data)
    query_text = query_file.read_text(encoding="utf-8")
    hits = run_query(lang, query_text, tree, data)
    if named_only:
        hits = [h for h in hits if not h["kind"].startswith("[")]
    typer.echo(json.dumps(hits, indent=2, ensure_ascii=False))


@app.command("symbols")
def symbols(dirpath: Path) -> None:
    """List Python symbol captures for an entire directory tree."""
    langs = load_langs()
    out: list[dict[str, object]] = []
    query_path = Path(__file__).resolve().parents[1] / "queries" / "python.scm"
    query_lines = query_path.read_text(encoding="utf-8").splitlines()
    query_text = "\n".join(
        line for line in query_lines if "def.name" in line or "def.params" in line
    )
    for candidate in dirpath.rglob("*.py"):
        data = candidate.read_bytes()
        tree = parse_bytes(langs.py, data)
        captures = run_query(langs.py, query_text, tree, data)
        out.append({"file": str(candidate), "defs": captures})
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
