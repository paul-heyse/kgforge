# src/kgfoundry_common/http/__init__.py
from pathlib import Path

from .client import HttpClient, HttpSettings
from .policy import PolicyRegistry
from .tenacity_retry import TenacityRetryStrategy


def make_client_with_policy(
    service: str, base_url: str, policy_name: str, policies_root: Path
) -> HttpClient:
    reg = PolicyRegistry(policies_root)
    pol = reg.get(policy_name)
    return HttpClient(
        settings=HttpSettings(service=service, base_url=base_url),
        retry_strategy=TenacityRetryStrategy(pol),
    )
