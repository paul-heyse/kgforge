# src/kgfoundry_common/http/client.py (excerpt)
from .types import RetryStrategy
from .tenacity_retry import TenacityRetryStrategy
from .policy import PolicyRegistry
from .errors import *

class HttpClient:
    def __init__(self, settings: HttpSettings, retry_strategy: RetryStrategy | None = None, ...):
        self.s = settings
        self.retry_strategy = retry_strategy  # may be None for single-attempt
        ...

    def _policy_strategy_for(self, method: str) -> RetryStrategy | None:
        if self.retry_strategy is None:
            return None
        # If strategy supports per-method specialization:
        if hasattr(self.retry_strategy, "for_method"):
            return self.retry_strategy.for_method(method)
        return self.retry_strategy

    def request(...):
        method = method.upper()
        url = self._build_url(url)
        headers = self._merge_headers(headers)
        # Non-idempotent safeguard for policies requiring Idempotency-Key
        if (method not in {"GET","HEAD","OPTIONS"} and
            isinstance(self.retry_strategy, TenacityRetryStrategy) and
            self.retry_strategy.policy.require_idempotency_key and
            "Idempotency-Key" not in headers):
            # Force single attempt for safety:
            strategy = None
        else:
            strategy = self._policy_strategy_for(method)

        def _attempt():
            # one attempt, raise well-typed HttpError on failure
            resp = self.b.request(method, url, params=params, headers=headers, json=json, data=data, timeout_s=timeout_s or self.s.read_timeout_s)
            status = resp.status_code
            if 200 <= status < 300:
                return resp
            body_excerpt = (lambda t: t[:500] if t else "")(resp.text())
            if status == 429:
                raise HttpRateLimited(status, body_excerpt, headers=resp.headers)
            raise HttpStatusError(status, body_excerpt, headers=resp.headers)

        if strategy is None:
            return _attempt()
        return strategy.run(_attempt)
