"""HTTP client with retry strategy support.

This package provides an HttpClient class that supports configurable retry strategies, idempotency
key requirements, and policy-based retry configuration.
"""

from __future__ import annotations

from pathlib import Path

from kgfoundry_common.http.client import HttpClient, HttpSettings
from kgfoundry_common.http.policy import PolicyRegistry
from kgfoundry_common.http.tenacity_retry import TenacityRetryStrategy


def make_client_with_policy(
    service: str, base_url: str, policy_name: str, policies_root: Path
) -> HttpClient:
    """Create HTTP client with retry policy loaded from file.

    Parameters
    ----------
    service : str
        Service name for logging and metrics.
    base_url : str
        Base URL for all requests.
    policy_name : str
        Name of retry policy to load (without .yaml extension).
    policies_root : Path
        Directory containing policy YAML files.

    Returns
    -------
    HttpClient
        Configured HTTP client instance.
    """
    reg = PolicyRegistry(policies_root)
    pol = reg.get(policy_name)
    return HttpClient(
        settings=HttpSettings(service=service, base_url=base_url),
        retry_strategy=TenacityRetryStrategy(pol),
    )
