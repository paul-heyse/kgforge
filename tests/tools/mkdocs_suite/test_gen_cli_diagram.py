from __future__ import annotations

import contextlib
import importlib
import io
from typing import TYPE_CHECKING

from tools.mkdocs_suite.docs.cli_diagram import collect_operations, write_diagram

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pytest


@contextlib.contextmanager
def _capture_write(
    buffers: dict[str, io.StringIO], path: str, mode: str
) -> Iterator[io.StringIO]:
    if "w" in mode:
        buffer = io.StringIO()
        buffers[path] = buffer
    else:
        buffer = buffers.setdefault(path, io.StringIO())
    yield buffer


def test_write_diagram_emits_single_node_for_multi_tag_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen_cli_module = importlib.import_module(
        "tools.mkdocs_suite.docs._scripts.gen_cli_diagram"
    )
    spec = {
        "paths": {
            "/cli/run": {
                "post": {
                    "operationId": "runCliCommand",
                    "summary": "Execute the CLI",
                    "tags": ["ingest", "admin", "ingest"],
                }
            }
        }
    }
    operations = collect_operations(spec)
    assert operations == [
        ("POST", "/cli/run", "runCliCommand", "Execute the CLI", ("ingest", "admin"))
    ]

    buffers: dict[str, io.StringIO] = {}

    def fake_open(
        path: str, mode: str = "r", **_: object
    ) -> contextlib.AbstractContextManager[io.StringIO]:
        return _capture_write(buffers, path, mode)

    monkeypatch.setattr(gen_cli_module.mkdocs_gen_files, "open", fake_open)

    write_diagram(operations)

    d2_output = buffers["diagrams/cli_by_tag.d2"].getvalue()

    assert d2_output.count('  "POST /cli/run":') == 1
    assert '  "ingest" -> "POST /cli/run"\n' in d2_output
    assert '  "admin" -> "POST /cli/run"\n' in d2_output
