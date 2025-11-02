## Summary

Eliminate all remaining `pyrefly` suppressions by standardising typed facades across observability, FAISS/cuVS adapters, FastAPI wiring, agent catalog math, and documentation tooling. The change introduces central modules (`tools._shared.prometheus`, `kgfoundry_common.numpy_typing`, `src.search_api.fastapi_helpers`) plus governance automation (`scripts/check_pyrefly_suppressions.py`) so every optional dependency path remains statically analysed and regression-proof.

## Public API Sketch

- `tools._shared.observability_facade`
  - `def log_operation(
        logger: logging.Logger,
        *,
        operation: str,
        status: Literal["ok", "error"],
        duration_ms: float,
        correlation_id: str | None = None,
        **extra: object,
    ) -> None`
  - `def record_counter(name: str, documentation: str, labels: Mapping[str, str] | None = None) -> CounterLike`
  - `def record_histogram(
        name: str,
        documentation: str,
        *,
        buckets: Sequence[float] | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> HistogramLike`
- `kgfoundry_common.numpy_typing`
  - Type aliases `FloatMatrix[T_contra: float | np.float64]`, `IntVector`, protocols for vector operations.
  - Functions `normalize_l2(v: np.ndarray) -> np.ndarray`, `topk_indices(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]`, `safe_argpartition(values: NDArray[np.float64], k: int) -> NDArray[np.int64]`.
- `src.search_api.fastapi_helpers`
  - `def typed_middleware(handler: MiddlewareCallable) -> MiddlewareCallable`
  - `def typed_dependency(factory: DependencyCallable[_T]) -> DependencyCallable[_T]`
  - `def typed_exception_handler(
        exc_type: type[Exception],
        handler: ExceptionHandlerCallable,
    ) -> ExceptionHandlerCallable`
- `scripts.check_pyrefly_suppressions`
  - CLI entry point `def main(argv: Sequence[str] | None = None) -> int` scanning for `type: ignore` usages lacking ticket references.

## Data/Schema Contracts

- Update `schema/examples/problem_details/*.json` to describe new FAISS/cuVS fallback errors and FastAPI Problem Details surfaces; ensure RFC 9457 compliance.
- Extend `schema/agents/catalog/*.json` (vector metadata) to reference new typed helper guarantees and document backward-compatible vector size constraints.
- Provide JSON Schema snippets in `tools/_shared/observability_facade.md` describing metric payloads and structured logging extras expected by tooling.
- Validate DuckDB ingestion schemas referenced by `docs/_scripts/shared.py` and surface `DocGenSettings` contract via JSON Schema 2020-12.

## Test Plan

- Static analysis gates: `uv run ruff check --fix` on touched packages; `uv run pyrefly check` and `uv run mypy --config-file mypy.ini` across `tools`, `src/search_api`, `src/vectorstore_faiss`, `src/kgfoundry_common`, `docs/_scripts`, and `site/_scripts`.
- Table-driven pytest modules covering FAISS GPU toggles, FastAPI exception wiring, and agent catalog vector helpers; add doctest/xdoctest snippets for new observability and NumPy helper APIs.
- Validate Problem Details and schema changes via existing schema lint tooling plus new round-trip tests for `DocGenSettings` serialisation.
- Run `scripts/check_pyrefly_suppressions.py` in CI and locally to confirm zero unmanaged suppressions; ensure failure path exercised in unit tests with temporary files.


