# src/kgfoundry_common/http/tenacity_retry.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tenacity import Retrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_base

from .errors import HttpStatusError
from .policy import RetryPolicyDoc
from .types import RetryStrategy


@dataclass
class WaitRetryAfterOrJitter(wait_base):
    initial: float
    max_s: float
    jitter: float  # 0..1 fraction
    base: float
    respect_retry_after: bool

    def __call__(self, retry_state) -> float:
        # attempt starts at 1
        attempt = retry_state.attempt_number
        # If last exception had Retry-After, prefer it:
        sleep = None
        if self.respect_retry_after:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if isinstance(exc, HttpStatusError):
                ra = _parse_retry_after(exc.headers.get("Retry-After"))
                if ra is not None:
                    sleep = min(ra, self.max_s)
        if sleep is None:
            # Exponential backoff with jitter
            base = min(self.max_s, self.initial * (self.base ** (attempt - 1)))
            jitter = base * self.jitter
            sleep = max(0.0, base - jitter + _rand() * (2 * jitter))
        return sleep


def _parse_retry_after(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _rand() -> float:
    # separate for test seeding
    import random

    return random.random()


def _status_in_sets(status: int, sets: tuple[tuple[int, int] | int, ...]) -> bool:
    for x in sets:
        if isinstance(x, int) and status == x:
            return True
        if isinstance(x, tuple) and x[0] <= status <= x[1]:
            return True
    return False


def _should_retry_exception(method: str, policy: RetryPolicyDoc) -> Callable[[BaseException], bool]:
    allowed_methods = set(policy.methods)
    retry_exc_names = set(policy.retry_exceptions)
    give_up = set(policy.give_up_status)
    status_sets = policy.retry_status

    def _pred(e: BaseException) -> bool:
        if method.upper() not in allowed_methods:
            return False
        # idempotency guard for non-idempotent methods
        if policy.require_idempotency_key and method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            # Tenacity has no access to headers; enforce earlier in client (see below)
            return False
        # Exception-based rule
        name = e.__class__.__name__
        if name in retry_exc_names:
            return True
        # HTTP status-based rule
        if isinstance(e, HttpStatusError):
            if e.status in give_up:
                return False
            return _status_in_sets(e.status, status_sets)
        return False

    return _pred


class TenacityRetryStrategy(RetryStrategy):
    def __init__(self, policy: RetryPolicyDoc):
        self.policy = policy

    def run(self, fn: Callable[[], object]):
        retry = retry_if_exception(_should_retry_exception(method="*UNKNOWN*", policy=self.policy))
        # note: we'll substitute actual method per-request (see client below)
        stopper = stop_after_attempt(self.policy.stop_after_attempt)
        if self.policy.stop_after_delay_s:
            stopper = stopper | stop_after_delay(self.policy.stop_after_delay_s)
        waiter = WaitRetryAfterOrJitter(
            initial=self.policy.wait_initial_s,
            max_s=self.policy.wait_max_s,
            jitter=self.policy.wait_jitter,
            base=self.policy.wait_base,
            respect_retry_after=self.policy.respect_retry_after,
        )
        # We'll set reraise=True so we get the final exception
        r = Retrying(retry=retry, stop=stopper, wait=waiter, reraise=True)
        return r(fn)

    def for_method(self, method: str) -> TenacityRetryStrategy:
        # clone with method-specific predicate
        pred = retry_if_exception(_should_retry_exception(method=method, policy=self.policy))
        stopper = stop_after_attempt(self.policy.stop_after_attempt)
        if self.policy.stop_after_delay_s:
            stopper = stopper | stop_after_delay(self.policy.stop_after_delay_s)
        waiter = WaitRetryAfterOrJitter(
            initial=self.policy.wait_initial_s,
            max_s=self.policy.wait_max_s,
            jitter=self.policy.wait_jitter,
            base=self.policy.wait_base,
            respect_retry_after=self.policy.respect_retry_after,
        )
        r = Retrying(retry=pred, stop=stopper, wait=waiter, reraise=True)

        # Wrap as a tiny strategy that uses this Retrying instance:
        class _MethodStrategy(RetryStrategy):
            def run(self, fn):
                return r(fn)

        return _MethodStrategy()
