"""
Provide utilities for module.

Notes
-----
This module exposes the primary interfaces for the package.

See Also
--------
observability.metrics
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
        """
        Represent NoopMetric.
        
        Attributes
        ----------
        None
            No public attributes documented.
        
        Methods
        -------
        labels()
            Method description.
        observe()
            Method description.
        inc()
            Method description.
        set()
            Method description.
        
        Examples
        --------
        >>> from observability.metrics import _NoopMetric
        >>> result = _NoopMetric()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        observability.metrics
        
        Notes
        -----
        Document class invariants and lifecycle details here.
        """
        
        

        def labels(self, *args: object, **kwargs: object) -> _NoopMetric:
            """
            Return labels.
            
            Parameters
            ----------
            *args : Any, optional
                Description for ``*args``.
            **kwargs : Any, optional
                Description for ``**kwargs``.
            
            Returns
            -------
            _NoopMetric
                Description of return value.
            
            Examples
            --------
            >>> from observability.metrics import labels
            >>> result = labels()
            >>> result  # doctest: +ELLIPSIS
            ...
            
            See Also
            --------
            observability.metrics
            
            Notes
            -----
            Provide usage considerations, constraints, or complexity notes.
            """
            
            return self

        def observe(self, *args: object, **kwargs: object) -> None:
            """
            Return observe.
            
            Parameters
            ----------
            *args : Any, optional
                Description for ``*args``.
            **kwargs : Any, optional
                Description for ``**kwargs``.
            
            Examples
            --------
            >>> from observability.metrics import observe
            >>> observe()  # doctest: +ELLIPSIS
            
            See Also
            --------
            observability.metrics
            
            Notes
            -----
            Provide usage considerations, constraints, or complexity notes.
            """
            
            return

        def inc(self, *args: object, **kwargs: object) -> None:
            """
            Return inc.
            
            Parameters
            ----------
            *args : Any, optional
                Description for ``*args``.
            **kwargs : Any, optional
                Description for ``**kwargs``.
            
            Examples
            --------
            >>> from observability.metrics import inc
            >>> inc()  # doctest: +ELLIPSIS
            
            See Also
            --------
            observability.metrics
            
            Notes
            -----
            Provide usage considerations, constraints, or complexity notes.
            """
            
            return

        def set(self, *args: object, **kwargs: object) -> None:
            """
            Return set.
            
            Parameters
            ----------
            *args : Any, optional
                Description for ``*args``.
            **kwargs : Any, optional
                Description for ``**kwargs``.
            
            Examples
            --------
            >>> from observability.metrics import set
            >>> set()  # doctest: +ELLIPSIS
            
            See Also
            --------
            observability.metrics
            
            Notes
            -----
            Provide usage considerations, constraints, or complexity notes.
            """
            
            return

    def _make_noop_metric(*args: object, **kwargs: object) -> _NoopMetric:
        """
        Return make noop metric.
        
        Parameters
        ----------
        *args : Any, optional
            Description for ``*args``.
        **kwargs : Any, optional
            Description for ``**kwargs``.
        
        Returns
        -------
        _NoopMetric
            Description of return value.
        
        Examples
        --------
        >>> from observability.metrics import _make_noop_metric
        >>> result = _make_noop_metric()
        >>> result  # doctest: +ELLIPSIS
        ...
        
        See Also
        --------
        observability.metrics
        
        Notes
        -----
        Provide usage considerations, constraints, or complexity notes.
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
