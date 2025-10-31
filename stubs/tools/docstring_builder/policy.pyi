from typing import Any

PolicyAction = Any
PolicyEngine = Any
PolicyException = Exception
PolicySettings = Any

def _apply_mapping(data: Any, *, logger: Any | None = ...) -> Any: ...
def _apply_overrides(data: Any, *, logger: Any | None = ...) -> Any: ...
def load_policy_settings(path: Any | None = ...) -> PolicySettings: ...
