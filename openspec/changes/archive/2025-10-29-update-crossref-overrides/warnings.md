# Cross-reference Warning Inventory

This scratch document tracks the `ref.*` warnings observed in `sphinx-warn.log`.

## Third-party annotations
- [x] `numpy.typing.NDArray` — normalized via `QUALIFIED_NAME_OVERRIDES` with fallback `nitpick_ignore` for the base alias.
- [x] `numpy.float32` / `np.float32`
- [x] `np.int64`
- [x] `np.str_`
- [x] `typer.Argument` / defaults emitted as `default=typer.Argument(...)`
- [x] `typer.Option` / defaults emitted as `default=typer.Option(...)`
- [x] `typer.Exit`
- [x] FastAPI dependencies: `HTTPException`, `Depends`, `Header`
- [x] `duckdb.DuckDBPyConnection` (present in proposal; confirm coverage once overrides expand)

## Internal aliases lacking targets
- [x] `VecArray` (and `src.search_api.faiss_adapter.VecArray`)
- [x] `FloatArray`
- [x] `StrArray`
- [x] `_SupportsHttp`
- [x] `_SupportsResponse`
- [x] `Id`
- [x] `Concept`
- [x] `Stability`

## Default value literals treated as types
- [x] Values such as `default=None`, `default=10`, `default='/data'`
- [x] Typer metadata strings (`help='...'`, `'lucene'`, `'CPU .idx'`, etc.)
- [x] FastAPI defaults (`default=Depends`, `default=Header`)

## Miscellaneous
- [x] Gallery quickstart references `kgfoundry` module (missing `py:mod` target)
- [ ] Duplicate target warning for `NDArray[np.float32]` between FAISS adapter pages

## Decisions (Task 1.3)
- [x] Prefer `QUALIFIED_NAME_OVERRIDES` for NumPy/PyArrow/DuckDB/FastAPI/Typer types; add missing short aliases (`np.str_`) and direct mappings for FastAPI defaults.
- [x] Use updated docstring generator to render default literals as inline code, avoiding mis-detected type references.
- [x] Add nav anchors and, where necessary, explicit exports so aliases like `VecArray`, `FloatArray`, `StrArray`, `Stability`, and `_SupportsHttp` have canonical documentation targets.
- [x] For Typer/FastAPI dependencies without upstream inventories, fall back to local canonical targets plus `nitpick_ignore` for cases we cannot resolve after overrides.
- [x] Update gallery quickstart to reference `src.kgfoundry` so the module lookup succeeds.
- [ ] Investigate and resolve duplicate `NDArray[np.float32]` declarations by ensuring each alias reports a unique canonical path.
