# src/kgfoundry_common/http/policy.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RetryPolicyDoc:
    name: str
    description: str | None
    methods: tuple[str, ...]
    retry_status: tuple[tuple[int, int] | int, ...]  # e.g., (500,504) or 429
    retry_exceptions: tuple[str, ...]
    give_up_status: tuple[int, ...]
    respect_retry_after: bool
    require_idempotency_key: bool
    stop_after_attempt: int
    stop_after_delay_s: float | None
    wait_kind: str
    wait_initial_s: float
    wait_max_s: float
    wait_jitter: float
    wait_base: float
    metrics_label: str | None


def _parse_status_entry(x: int | str) -> tuple[int, int] | int:
    if isinstance(x, int):
        return x
    lo, hi = x.split("-", 1)
    return (int(lo), int(hi))


def load_policy(path: Path, schema_path: Path | None = None) -> RetryPolicyDoc:
    obj = yaml.safe_load(path.read_text())
    if schema_path and schema_path.exists():
        import jsonschema

        jsonschema.validate(obj, json.loads(schema_path.read_text()))
    return RetryPolicyDoc(
        name=obj["name"],
        description=obj.get("description"),
        methods=tuple(m.upper() for m in obj["methods"]),
        retry_status=tuple(_parse_status_entry(s) for s in obj["retry_on"].get("status", [])),
        retry_exceptions=tuple(obj["retry_on"].get("exceptions", [])),
        give_up_status=tuple(obj.get("give_up_on_status", [])),
        respect_retry_after=bool(obj.get("respect_retry_after", False)),
        require_idempotency_key=bool(obj.get("require_idempotency_key", False)),
        stop_after_attempt=int(obj["stop"]["after_attempt"]),
        stop_after_delay_s=float(obj["stop"].get("after_delay_s"))
        if obj["stop"].get("after_delay_s") is not None
        else None,
        wait_kind=obj["wait"]["kind"],
        wait_initial_s=float(obj["wait"]["initial_s"]),
        wait_max_s=float(obj["wait"]["max_s"]),
        wait_jitter=float(obj["wait"]["jitter"]),
        wait_base=float(obj["wait"].get("base", 2.0)),
        metrics_label=obj.get("metrics_label"),
    )


class PolicyRegistry:
    def __init__(self, root: Path):
        self.root = root

    def get(self, name: str) -> RetryPolicyDoc:
        p = self.root / f"{name}.yaml"
        if not p.exists():
            raise FileNotFoundError(p)
        schema = Path(__file__).with_name("policy.schema.json")
        return load_policy(p, schema)
