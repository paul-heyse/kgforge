"""Type definitions for HTTP client retry strategies.

This module defines the RetryStrategy protocol that all retry implementations must conform to.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class RetryStrategy(Protocol[T]):
    """Protocol for retry strategies.

    Implementations of this protocol execute a callable with retry logic according to a configured
    policy, re-raising the final error if all retries are exhausted.
    """

    def run(self, fn: Callable[[], T]) -> T:
        """Execute fn with retries per configured policy; re-raise final error.

        Parameters
        ----------
        fn : Callable[[], T]
            The function to execute with retry logic.

        Returns
        -------
        T
            The result of fn if execution succeeds.

        Notes
        -----
        Implementations should re-raise the final exception if all retries
        are exhausted. The specific exception type depends on what the
        wrapped function raises.
        """
        ...
