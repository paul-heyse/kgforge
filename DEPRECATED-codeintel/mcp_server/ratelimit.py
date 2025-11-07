"""Token bucket rate limiter for MCP server."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    """Simple token bucket rate limiter for controlling request throughput.

    This class implements a token bucket algorithm to enforce rate limits on
    operations. Tokens are refilled at a constant rate (tokens per second), and
    operations consume tokens. If sufficient tokens are available, the operation
    proceeds; otherwise, it is rate-limited.

    The bucket has a maximum capacity (burst size) that allows short bursts of
    activity above the sustained rate. This is useful for MCP server operations
    where occasional bursts are acceptable but sustained high rates should be
    throttled.

    Attributes
    ----------
    rate : float
        Tokens per second refill rate. This determines the sustained throughput
        allowed by the rate limiter.
    burst : int
        Maximum token capacity (burst size). This allows short bursts above the
        sustained rate before throttling kicks in.
    tokens : float
        Current token count in the bucket, by default 0.0. Tokens are refilled
        based on elapsed time since the last update.
    last : float
        Last update timestamp (monotonic time), by default current time. Used
        to calculate elapsed time for token refill.

    Examples
    --------
    >>> bucket = TokenBucket(rate=5.0, burst=10)
    >>> if bucket.acquire():
    ...     # Operation allowed
    ...     pass
    >>> # After 1 second, 5 tokens will be refilled
    """

    rate: float  # tokens per second
    burst: int
    tokens: float = 0.0
    last: float = time.monotonic()

    def __post_init__(self) -> None:
        """Initialise the bucket with a full burst to avoid cold-start throttling."""
        if self.tokens <= 0:
            self.tokens = float(self.burst)

    def acquire(self, n: float = 1.0) -> bool:
        """Attempt to acquire tokens from the bucket.

        Parameters
        ----------
        n : float, optional
            Number of tokens to acquire, by default 1.0.

        Returns
        -------
        bool
            True if tokens were acquired, False if rate limit exceeded.
        """
        now = time.monotonic()
        elapsed = now - self.last
        # Refill tokens based on elapsed time
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
