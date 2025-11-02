"""Typed Prometheus helpers with graceful fallbacks.

The helpers in this module consolidate optional Prometheus dependencies behind
typed constructors so call sites never need to sprinkle ``type: ignore``
pragmas. When :mod:`prometheus_client` is unavailable the helpers return
lightweight no-op implementations that honour the same interface.

Examples
--------
Create metrics when Prometheus is installed:

>>> from kgfoundry_common.prometheus import build_counter
>>> counter = build_counter("example_total", "Example operations", ["status"])
>>> counter.labels(status="success").inc()

Fallback behaviour (no Prometheus import required):

>>> from unittest.mock import patch
>>> import kgfoundry_common.prometheus as prom
>>> with patch("kgfoundry_common.prometheus._COUNTER_CONSTRUCTOR", None):
...     noop = prom.build_counter("noop_total", "Noop counter")
...     noop.inc()
...     noop.labels(status="success").inc()
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn, Protocol, cast, overload

__all__ = [
    "HAVE_PROMETHEUS",
    "CollectorRegistry",
    "CounterLike",
    "GaugeLike",
    "HistogramLike",
    "HistogramParams",
    "build_counter",
    "build_gauge",
    "build_histogram",
    "get_default_registry",
]


if TYPE_CHECKING:  # pragma: no cover - typing only
    from prometheus_client import Counter as _PromCounterType
    from prometheus_client import Gauge as _PromGaugeType
    from prometheus_client import Histogram as _PromHistogramType
    from prometheus_client.registry import CollectorRegistry
else:  # pragma: no cover - runtime fallback when dependency missing

    class _PromCounterType:
        """Runtime stub matching Prometheus counter surface."""

        def labels(self, **labels: object) -> _PromCounterType:
            """Return the counter stub regardless of label input."""
            del labels
            return self

        def inc(self, value: float = 1.0) -> None:
            """Perform no operation when Prometheus is absent."""
            _ = self
            del value

    class _PromGaugeType:
        """Runtime stub matching Prometheus gauge surface."""

        def labels(self, **labels: object) -> _PromGaugeType:
            """Return the gauge stub regardless of label input."""
            del labels
            return self

        def set(self, value: float) -> None:
            """Perform no operation when Prometheus is absent."""
            _ = self
            del value

    class _PromHistogramType:
        """Runtime stub matching Prometheus histogram surface."""

        def labels(self, **labels: object) -> _PromHistogramType:
            """Return the histogram stub regardless of label input."""
            del labels
            return self

        def observe(self, value: float) -> None:
            """Perform no operation when Prometheus is absent."""
            _ = self
            del value

    CollectorRegistry = object


class CounterLike(Protocol):
    """Protocol describing Prometheus counter behaviour relied upon."""

    def labels(self, **labels: object) -> CounterLike:
        """Return a counter labelled with the provided fields."""
        ...

    def inc(self, value: float = 1.0) -> None:
        """Increment the counter by ``value``."""
        ...


class GaugeLike(Protocol):
    """Protocol describing Prometheus gauge behaviour relied upon."""

    def labels(self, **labels: object) -> GaugeLike:
        """Return a gauge labelled with the provided fields."""
        ...

    def set(self, value: float) -> None:
        """Set the gauge to ``value``."""
        ...


class HistogramLike(Protocol):
    """Protocol describing Prometheus histogram behaviour relied upon."""

    def labels(self, **labels: object) -> HistogramLike:
        """Return a histogram labelled with the provided fields."""
        ...

    def observe(self, value: float) -> None:
        """Record an observation of ``value``."""
        ...


class _CounterConstructor(Protocol):
    def __call__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        registry: CollectorRegistry | None = ...,
        unit: str | None = ...,
        **kwargs: object,
    ) -> CounterLike: ...


class _GaugeConstructor(Protocol):
    def __call__(
        self,
        name: str,
        documentation: str,
        labelnames: Sequence[str] | None = ...,
        *,
        registry: CollectorRegistry | None = ...,
        unit: str | None = ...,
        **kwargs: object,
    ) -> GaugeLike: ...


class _HistogramConstructor(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> HistogramLike: ...


@dataclass(slots=True)
class HistogramParams:
    """Configuration for building a histogram metric."""

    name: str
    documentation: str
    labelnames: Sequence[str] | None = None
    buckets: Sequence[float] | None = None
    registry: CollectorRegistry | None = None
    unit: str | None = None


class _NoopCounter:
    """Counter stub used when Prometheus is unavailable."""

    __slots__ = ()

    def labels(self, **labels: object) -> _NoopCounter:
        """Return the stub counter regardless of label input."""
        del labels
        return self

    def inc(self, value: float = 1.0) -> None:
        """Perform no operation when Prometheus is absent."""
        _ = self
        del value


class _NoopGauge:
    """Gauge stub used when Prometheus is unavailable."""

    __slots__ = ()

    def labels(self, **labels: object) -> _NoopGauge:
        """Return the stub gauge regardless of label input."""
        del labels
        return self

    def set(self, value: float) -> None:
        """Perform no operation when Prometheus is absent."""
        _ = self
        del value


class _NoopHistogram:
    """Histogram stub used when Prometheus is unavailable."""

    __slots__ = ()

    def labels(self, **labels: object) -> _NoopHistogram:
        """Return the stub histogram regardless of label input."""
        del labels
        return self

    def observe(self, value: float) -> None:
        """Perform no operation when Prometheus is absent."""
        _ = self
        del value


_COUNTER_CONSTRUCTOR: _CounterConstructor | None = None
_GAUGE_CONSTRUCTOR: _GaugeConstructor | None = None
_HISTOGRAM_CONSTRUCTOR: _HistogramConstructor | None = None
_DEFAULT_REGISTRY: object | None = None
_PROMETHEUS_VERSION: str | None = None


try:  # pragma: no cover - exercised in environments with Prometheus installed
    import prometheus_client
    from prometheus_client import REGISTRY as _PROMETHEUS_REGISTRY
    from prometheus_client import Counter as _PromCounter
    from prometheus_client import Gauge as _PromGauge
    from prometheus_client import Histogram as _PromHistogram
    from prometheus_client.registry import CollectorRegistry

except ImportError:  # pragma: no cover - handled by fallback
    HAVE_PROMETHEUS = False
else:  # pragma: no cover - exercised when dependency present
    HAVE_PROMETHEUS = True
    _COUNTER_CONSTRUCTOR = cast(_CounterConstructor, _PromCounter)
    _GAUGE_CONSTRUCTOR = cast(_GaugeConstructor, _PromGauge)
    _HISTOGRAM_CONSTRUCTOR = cast(_HistogramConstructor, _PromHistogram)
    _DEFAULT_REGISTRY = _PROMETHEUS_REGISTRY
    _PROMETHEUS_VERSION = getattr(prometheus_client, "__version__", None)


def _labels_or_default(labelnames: Sequence[str] | None) -> Sequence[str]:
    return tuple(labelnames) if labelnames is not None else ()


_MAX_HISTOGRAM_POSITIONAL_ARGS = 2
_DOCUMENTATION_POSITION = 1


def _histogram_type_error(message: str) -> NoReturn:
    raise TypeError(message)


def build_counter(
    name: str,
    documentation: str,
    labelnames: Sequence[str] | None = None,
    *,
    registry: CollectorRegistry | None = None,
    unit: str | None = None,
) -> CounterLike:
    """Return a counter metric or a no-op stub.

    Parameters
    ----------
    name : str
        Metric name registered with Prometheus.
    documentation : str
        Human readable description of the metric.
    labelnames : Sequence[str] | None, optional
        Label names applied to the metric (defaults to empty tuple).
    registry : CollectorRegistry | None, optional
        Prometheus registry to register against (defaults to global registry).
    unit : str | None, optional
        Unit description recorded alongside the metric.
    """
    constructor = _COUNTER_CONSTRUCTOR
    if constructor is None:
        return _NoopCounter()
    return constructor(
        name,
        documentation,
        _labels_or_default(labelnames),
        registry=registry,
        unit=unit,
    )


def build_gauge(
    name: str,
    documentation: str,
    labelnames: Sequence[str] | None = None,
    *,
    registry: CollectorRegistry | None = None,
    unit: str | None = None,
) -> GaugeLike:
    """Return a gauge metric or a no-op stub."""
    constructor = _GAUGE_CONSTRUCTOR
    if constructor is None:
        return _NoopGauge()
    return constructor(
        name,
        documentation,
        _labels_or_default(labelnames),
        registry=registry,
        unit=unit,
    )


def _coerce_histogram_params(*args: object, **kwargs: object) -> HistogramParams:
    """Normalize legacy histogram arguments into a :class:`HistogramParams` instance."""
    if args and isinstance(args[0], HistogramParams):
        if len(args) > 1 or kwargs:
            _histogram_type_error("build_histogram() received unexpected extra arguments")
        return args[0]

    if not args:
        _histogram_type_error("build_histogram() missing required argument: 'name'")

    if len(args) > _MAX_HISTOGRAM_POSITIONAL_ARGS:
        _histogram_type_error("build_histogram() received too many positional arguments")

    name = str(args[0])
    if len(args) > _DOCUMENTATION_POSITION:
        documentation = str(args[_DOCUMENTATION_POSITION])
    else:
        doc = kwargs.pop("documentation", None)
        if doc is None:
            _histogram_type_error("build_histogram() missing required argument: 'documentation'")
        documentation = str(doc)

    labelnames = kwargs.pop("labelnames", None)
    buckets = kwargs.pop("buckets", None)
    registry = kwargs.pop("registry", None)
    unit = kwargs.pop("unit", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        _histogram_type_error(f"build_histogram() got unexpected keyword arguments: {unexpected}")

    return HistogramParams(
        name=name,
        documentation=documentation,
        labelnames=cast(Sequence[str] | None, labelnames),
        buckets=cast(Sequence[float] | None, buckets),
        registry=cast(CollectorRegistry | None, registry),
        unit=None if unit is None else str(unit),
    )


@overload
def build_histogram(params: HistogramParams) -> HistogramLike: ...


@overload
def build_histogram(
    name: str,
    documentation: str,
    labelnames: Sequence[str] | None = ...,
    *,
    buckets: Sequence[float] | None = ...,
    registry: CollectorRegistry | None = ...,
    unit: str | None = ...,
) -> HistogramLike: ...


def build_histogram(*args: object, **kwargs: object) -> HistogramLike:
    """Return a histogram metric or a no-op stub."""
    constructor = _HISTOGRAM_CONSTRUCTOR
    if constructor is None:
        return _NoopHistogram()

    params = _coerce_histogram_params(*args, **kwargs)
    label_tuple = _labels_or_default(params.labelnames)
    call_kwargs: dict[str, object] = {}
    if params.registry is not None:
        call_kwargs["registry"] = params.registry
    if params.unit is not None:
        call_kwargs["unit"] = params.unit
    if params.buckets is not None:
        call_kwargs["buckets"] = tuple(params.buckets)

    return constructor(
        params.name,
        params.documentation,
        label_tuple,
        **call_kwargs,
    )


def get_default_registry() -> object | None:
    """Return the global Prometheus registry when the dependency is available."""
    return _DEFAULT_REGISTRY


def prometheus_version() -> str | None:
    """Return the detected :mod:`prometheus_client` version for diagnostics."""
    return _PROMETHEUS_VERSION
