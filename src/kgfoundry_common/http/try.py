# src/kgfoundry_common/http/types.py
from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class RetryStrategy(Protocol):
    def run(self, fn: Callable[[], T]) -> T:
        """Execute fn with retries per configured policy; re-raise final error."""
