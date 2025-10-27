"""Module for observability.metrics."""

from typing import Dict

try:
    from prometheus_client import Counter, Histogram, Gauge  # type: ignore
except Exception:  # pragma: no cover - minimal no-op fallbacks
    class _Noop:
        """Minimal stand-in when Prometheus client is unavailable."""

        def labels(self, *args, **kwargs):
            """Return self to mimic the Prometheus API."""
            return self

        def observe(self, *args, **kwargs):
            """Ignore observation calls."""

            return None

        def inc(self, *args, **kwargs):
            """Ignore counter increments."""

            return None

        def set(self, *args, **kwargs):
            """Ignore gauge updates."""

            return None

    Counter = Histogram = Gauge = _Noop  # type: ignore

pdf_download_success_total = Counter('pdf_download_success_total', 'Successful OA PDF downloads')
pdf_download_failure_total = Counter('pdf_download_failure_total', 'Failed OA PDF downloads', ['reason'])
search_total_latency_ms = Histogram('search_total_latency_ms', 'End-to-end /search latency (ms)')
faiss_search_latency_ms = Histogram('faiss_search_latency_ms', 'FAISS search latency (ms)')
bm25_queries_total = Counter('bm25_queries_total', 'BM25 queries issued')
splade_queries_total = Counter('splade_queries_total', 'SPLADE queries issued')
