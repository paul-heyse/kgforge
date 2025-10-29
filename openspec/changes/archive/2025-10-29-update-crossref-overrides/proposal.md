## Why
Sphinx is still emitting dozens of cross-reference warnings whenever we build the docs. The
most common issues are unresolved type annotations (``numpy.typing.NDArray``,
``pyarrow.schema``, ``duckdb.DuckDBPyConnection``), missing anchors for our own type
aliases (``VecArray``, ``StrArray``), and duplicate targets for re-exported exceptions
(``DownloadError``). These warnings create broken links in the rendered documentation and
prevent us from enabling strict “warnings-as-errors” builds.

## What Changes
- Expand the `QUALIFIED_NAME_OVERRIDES` mapping and/or intersphinx configuration so the most
  common third-party annotations resolve automatically.
- Ensure our internal aliases (e.g., ``VecArray``, ``StrArray``, ``Concept``) expose a single
  canonical target with appropriate ``[nav:anchor]`` markers.
- Normalise exception exports so ``DownloadError`` and ``UnsupportedMIMEError`` each have a
  single authoritative module; other modules import+re-export instead of defining duplicates.
- Document the updated mapping strategy so future contributors know how to add new type
  overrides without guessing.

## Impact
- Affected specs: none (tooling/documentation quality improvement).
- Affected code/config:
  - `tools/auto_docstrings.py` (``QUALIFIED_NAME_OVERRIDES``)
  - `docs/conf.py` (``intersphinx_mapping`` / ``nitpick_ignore`` refinements)
  - Source modules that re-export shared exceptions or type aliases
  - Tests or CI scripts if we add coverage for the cross-reference checks
- Expected outcome: Running `tools/update_docs.sh` yields zero ``ref.*`` warnings for the
  covered annotations, and rendered HTML includes working hyperlinks for these types.

