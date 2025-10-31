"""Overview of metrics.

This module bundles metrics logic for the kgfoundry stack. It groups related helpers so downstream
packages can import a single cohesive namespace. Refer to the functions and classes below for
implementation specifics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, cast

_TMetric = TypeVar("_TMetric", covariant=True)


class MetricFactory(Protocol[_TMetric]):
    def __call__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        registry: object | None = ...,
        **kwargs: object,
    ) -> _TMetric: ...


if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from prometheus_client import Counter as _CounterType
    from prometheus_client import Gauge as _GaugeType
    from prometheus_client import Histogram as _HistogramType
else:  # pragma: no cover - runtime fallback types

    class _CounterType:
        """Runtime stub when prometheus_client is unavailable."""

    class _GaugeType:
        """Runtime stub when prometheus_client is unavailable."""

    class _HistogramType:
        """Runtime stub when prometheus_client is unavailable."""


CounterFactory: TypeAlias = MetricFactory[_CounterType]
GaugeFactory: TypeAlias = MetricFactory[_GaugeType]
HistogramFactory: TypeAlias = MetricFactory[_HistogramType]

_PromCounter: MetricFactory[_CounterType] | None = None
_PromGauge: MetricFactory[_GaugeType] | None = None
_PromHistogram: MetricFactory[_HistogramType] | None = None

try:
    from prometheus_client import Counter as _PROM_COUNTER_IMPL
    from prometheus_client import Gauge as _PROM_GAUGE_IMPL
    from prometheus_client import Histogram as _PROM_HISTOGRAM_IMPL

    _PromCounter = cast(MetricFactory[_CounterType], _PROM_COUNTER_IMPL)
    _PromGauge = cast(MetricFactory[_GaugeType], _PROM_GAUGE_IMPL)
    _PromHistogram = cast(MetricFactory[_HistogramType], _PROM_HISTOGRAM_IMPL)
except ImportError:  # pragma: no cover - optional dependency guard
    pass


if _PromCounter is None or _PromGauge is None or _PromHistogram is None:

    class _NoopMetric:
        """Describe  NoopMetric.

        &lt;!-- auto:docstring-builder v1 --&gt;

        Describe the data structure and how instances collaborate with the surrounding package. Highlight how the class supports nearby modules to guide readers through the codebase.
        """

        def labels(self, *args: object, **kwargs: object) -> _NoopMetric:
            """Describe labels.

            &lt;!-- auto:docstring-builder v1 --&gt;

            Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

            Parameters
            ----------
            *args : object, optional, by default ()
            Configure the args. Defaults to ``()``.
            **kwargs : object, optional, by default {}
            Configure the kwargs. Defaults to ``{}``.


            Returns
            -------
            _NoopMetric
            Describe return value.
            """
            del args, kwargs
            return self

        def observe(self, *args: object, **kwargs: object) -> None:
            """Describe observe.

            &lt;!-- auto:docstring-builder v1 --&gt;

            Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

            Parameters
            ----------
            *args : object, optional, by default ()
            Configure the args. Defaults to ``()``.
            **kwargs : object, optional, by default {}
            Configure the kwargs. Defaults to ``{}``.
            """
            del args, kwargs

        def inc(self, *args: object, **kwargs: object) -> None:
            """Describe inc.

            &lt;!-- auto:docstring-builder v1 --&gt;

            Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

            Parameters
            ----------
            *args : object, optional, by default ()
            Configure the args. Defaults to ``()``.
            **kwargs : object, optional, by default {}
            Configure the kwargs. Defaults to ``{}``.
            """
            del args, kwargs

        def set(self, *args: object, **kwargs: object) -> None:
            """Describe set.

            &lt;!-- auto:docstring-builder v1 --&gt;

            Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

            Parameters
            ----------
            *args : object, optional, by default ()
            Configure the args. Defaults to ``()``.
            **kwargs : object, optional, by default {}
            Configure the kwargs. Defaults to ``{}``.
            """
            del args, kwargs

    def _make_noop_metric(*args: object, **kwargs: object) -> _NoopMetric:
        """Describe  make noop metric.

        &lt;!-- auto:docstring-builder v1 --&gt;

        Special method customising Python&#39;s object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language&#39;s data model.

        Parameters
        ----------
        *args : object, optional, by default ()
        Configure the args. Defaults to ``()``.
        **kwargs : object, optional, by default {}
        Configure the kwargs. Defaults to ``{}``.


        Returns
        -------
        _NoopMetric
        Describe return value.
        """
        del args, kwargs
        return _NoopMetric()

    Counter = cast(CounterFactory, _make_noop_metric)
    Gauge = cast(GaugeFactory, _make_noop_metric)
    Histogram = cast(HistogramFactory, _make_noop_metric)
else:
    Counter = _PromCounter
    Gauge = _PromGauge
    Histogram = _PromHistogram

pdf_download_success_total = Counter("pdf_download_success_total", "Successful OA PDF downloads")
pdf_download_failure_total = Counter(
    "pdf_download_failure_total", "Failed OA PDF downloads", ["reason"]
)
search_total_latency_ms = Histogram("search_total_latency_ms", "End-to-end /search latency (ms)")
faiss_search_latency_ms = Histogram("faiss_search_latency_ms", "FAISS search latency (ms)")
bm25_queries_total = Counter("bm25_queries_total", "BM25 queries issued")
splade_queries_total = Counter("splade_queries_total", "SPLADE queries issued")
