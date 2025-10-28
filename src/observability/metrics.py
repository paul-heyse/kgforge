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
        """Describe NoopMetric."""

        def labels(self, *args: object, **kwargs: object) -> _NoopMetric:
            """Compute labels.

            Carry out the labels operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
            
            Parameters
            ----------
            *args : object
            *args : object
                Description for ``*args``.
            **kwargs : object
            **kwargs : object
                Description for ``**kwargs``.
            
            Returns
            -------
            _NoopMetric
                Description of return value.
            
            Examples
            --------
            >>> from observability.metrics import labels
            >>> result = labels(*args, **kwargs)
            >>> result  # doctest: +ELLIPSIS
            ...
            """
            return self

        def observe(self, *args: object, **kwargs: object) -> None:
            """Compute observe.

            Carry out the observe operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

            Parameters
            ----------
            *args : object
            *args : object
                Description for ``*args``.
            **kwargs : object
            **kwargs : object
                Description for ``**kwargs``.

            Examples
            --------
            >>> from observability.metrics import observe
            >>> observe(*args, **kwargs)  # doctest: +ELLIPSIS
            """
            return

        def inc(self, *args: object, **kwargs: object) -> None:
            """Compute inc.

            Carry out the inc operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

            Parameters
            ----------
            *args : object
            *args : object
                Description for ``*args``.
            **kwargs : object
            **kwargs : object
                Description for ``**kwargs``.

            Examples
            --------
            >>> from observability.metrics import inc
            >>> inc(*args, **kwargs)  # doctest: +ELLIPSIS
            """
            return

        def set(self, *args: object, **kwargs: object) -> None:
            """Compute set.

            Carry out the set operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

            Parameters
            ----------
            *args : object
            *args : object
                Description for ``*args``.
            **kwargs : object
            **kwargs : object
                Description for ``**kwargs``.

            Examples
            --------
            >>> from observability.metrics import set
            >>> set(*args, **kwargs)  # doctest: +ELLIPSIS
            """
            return

    def _make_noop_metric(*args: object, **kwargs: object) -> _NoopMetric:
        """Compute make noop metric.

        Carry out the make noop metric operation.

        Parameters
        ----------
        *args : Any
            Description for ``*args``.
        **kwargs : Any
            Description for ``**kwargs``.

        Returns
        -------
        _NoopMetric
            Description of return value.
        """
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
