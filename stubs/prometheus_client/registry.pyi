from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

class Sample(NamedTuple):
    """A Prometheus metric sample.

    Attributes
    ----------
    name : str
        The metric name.
    labels : dict[str, str]
        Label dictionary mapping label names to values.
    value : float | int
        The metric value.
    timestamp : float | None
        Optional timestamp.
    exemplar : object
        Optional exemplar.
    native_histogram : object
        Optional native histogram.
    """

    name: str
    labels: dict[str, str]
    value: float | int
    timestamp: float | None
    exemplar: object
    native_histogram: object

class Metric:
    """A Prometheus metric (counter, gauge, histogram, etc.).

    Attributes
    ----------
    name : str
        The metric name.
    documentation : str
        Documentation for the metric.
    type : str
        The metric type ('counter', 'gauge', 'histogram', 'summary', 'untyped').
    samples : list[Sample]
        The metric samples.
    """

    name: str
    documentation: str
    type: str
    samples: list[Sample]

class CollectorRegistry:
    """Registry of Prometheus metrics."""

    def __init__(self, auto_describe: bool = ...) -> None: ...
    def register(self, metric: object) -> None: ...
    def unregister(self, metric: object) -> None: ...
    def collect(self) -> Iterable[Metric]: ...
