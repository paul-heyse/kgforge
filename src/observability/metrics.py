"""Overview of metrics.

This module bundles metrics logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from prometheus_client import Counter as _CounterType
    from prometheus_client import Gauge as _GaugeType
    from prometheus_client import Histogram as _HistogramType

    CounterFactory = Callable[..., _CounterType]
    GaugeFactory = Callable[..., _GaugeType]
    HistogramFactory = Callable[..., _HistogramType]
else:  # pragma: no cover - runtime fallback types
    CounterFactory = Callable[..., object]
    GaugeFactory = Callable[..., object]
    HistogramFactory = Callable[..., object]

try:
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Gauge as _PromGauge
    from prometheus_client import Histogram as _PromHistogram
except Exception:  # pragma: no cover - minimal no-op fallbacks

    class _NoopMetric:
        """No-op metric used when ``prometheus_client`` is unavailable."""

        def labels(self, *args: object, **kwargs: object) -> _NoopMetric:
            """Ignore label requests and return ``self`` for chaining."""
            return self

        def observe(self, *args: object, **kwargs: object) -> None:
            """Ignore observations when instrumentation is disabled."""
            return

        def inc(self, *args: object, **kwargs: object) -> None:
            """Ignore counter increments when instrumentation is disabled."""
            return

        def set(self, *args: object, **kwargs: object) -> None:
            """Ignore gauge updates when instrumentation is disabled."""
            return

    def _make_noop_metric(*args: object, **kwargs: object) -> _NoopMetric:
        """Return a no-op metric placeholder."""
        return _NoopMetric()

    Counter = cast(CounterFactory, _make_noop_metric)
    Gauge = cast(GaugeFactory, _make_noop_metric)
    Histogram = cast(HistogramFactory, _make_noop_metric)
else:
    Counter = cast(CounterFactory, _PromCounter)
    Gauge = cast(GaugeFactory, _PromGauge)
    Histogram = cast(HistogramFactory, _PromHistogram)

pdf_download_success_total = Counter("pdf_download_success_total", "Successful OA PDF downloads")
pdf_download_failure_total = Counter(
    "pdf_download_failure_total", "Failed OA PDF downloads", ["reason"]
)
search_total_latency_ms = Histogram("search_total_latency_ms", "End-to-end /search latency (ms)")
faiss_search_latency_ms = Histogram("faiss_search_latency_ms", "FAISS search latency (ms)")
bm25_queries_total = Counter("bm25_queries_total", "BM25 queries issued")
splade_queries_total = Counter("splade_queries_total", "SPLADE queries issued")
