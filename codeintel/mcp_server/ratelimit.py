"""Token bucket rate limiter for MCP server."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TokenBucket:
    """Simple token bucket rate limiter.

    Parameters
    ----------
    rate : float
        Tokens per second refill rate.
    burst : int
        Maximum token capacity (burst size).
    tokens : float, optional
        Current token count, by default 0.0.
    last : float, optional
        Last update timestamp (monotonic), by default current time.
    """

    rate: float  # tokens per second
    burst: int
    tokens: float = 0.0
    last: float = time.monotonic()

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
