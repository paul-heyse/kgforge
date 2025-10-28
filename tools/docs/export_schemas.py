#!/usr/bin/env python3
"""Export Pydantic & Pandera schemas with stable IDs, examples, and drift summaries.

Outputs:
  docs/reference/schemas/<module>.<Class>.json     # canonicalized, version-stamped, example-rich
  docs/_build/schema_drift.json                    # per-file drift summaries (only when drift exists)

CLI:
  --ref-template '#/$defs/{model}'   # Pydantic $ref template (default is Pydantic's default)
  --base-url 'https://kgfoundry/schemas'   # base for $id (env SCHEMA_BASE_URL overrides)
  --check-drift                      # do not write; fail (exit 2) if any drift is detected
  --by-alias                         # generate schema using field aliases for Pydantic models

Environment:
  SCHEMA_BASE_URL=https://kgfoundry/schemas
  DOCS_LINK_MODE=editor|github (unused here, consistent with pipeline)
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "reference" / "schemas"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

TOP_PACKAGES = (
    "kgfoundry",
    "kgfoundry_common",
    "kg_builder",
    "observability",
    "orchestration",
    "registry",
    "search_api",
)

NAVMAP = ROOT / "site" / "_build" / "navmap" / "navmap.json"
DRIFT_OUT = ROOT / "docs" / "_build" / "schema_drift.json"

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
DEFAULT_BASE_URL = os.getenv("SCHEMA_BASE_URL", "https://kgfoundry/schemas")


# --------------------------- utils ---------------------------


def _sorted(obj: Any) -> Any:
    """Recursively sort dict keys; leave arrays as-is for semantics."""
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sorted(x) for x in obj]
    return obj


def _write_if_changed(path: Path, data: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """Write JSON with sorted keys + trailing newline if content changed. Return (changed, old, new)."""
    new_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    old_text = path.read_text(encoding="utf-8") if path.exists() else None
    if old_text == new_text:
        return (False, old_text, new_text)
    path.write_text(new_text, encoding="utf-8")
    return (True, old_text, new_text)


def _module_iter() -> Iterable[str]:
    """Walk all subpackages of TOP_PACKAGES."""
    for pkg in TOP_PACKAGES:
        try:
            module = importlib.import_module(pkg)
        except Exception:
            continue
        if not hasattr(module, "__path__"):
            continue
        for info in pkgutil.walk_packages(module.__path__, prefix=f"{pkg}."):
            yield info.name


def is_pydantic_model(obj: object) -> bool:
    """Is pydantic model.

    Parameters
    ----------
    obj : object
        Description.

    Returns
    -------
    bool
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> is_pydantic_model(...)
    """
    try:
        from pydantic import BaseModel
    except Exception:
        return False
    return inspect.isclass(obj) and issubclass(obj, BaseModel)


def is_pandera_model(obj: object) -> bool:
    """Is pandera model.

    Parameters
    ----------
    obj : object
        Description.

    Returns
    -------
    bool
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> is_pandera_model(...)
    """
    try:
        import pandera as pa
    except Exception:
        return False
    schema_model = getattr(pa, "SchemaModel", None)
    return bool(schema_model and inspect.isclass(obj) and issubclass(obj, schema_model))


def _load_navmap() -> dict[str, Any]:
    """Load navmap.

    Returns
    -------
    dict[str, Any]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _load_navmap(...)
    """
    if not NAVMAP.exists():
        return {}
    try:
        return json.loads(NAVMAP.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _nav_versions(module_name: str, class_name: str, nav: dict[str, Any]) -> dict[str, Any] | None:
    """Find {x-version-introduced, x-deprecated-in} from navmap meta (module + symbol name match)."""
    mods = nav.get("modules") or {}
    entry = mods.get(module_name)
    if not entry:
        return None
    meta = (entry.get("meta") or {}).get(class_name) or {}
    out = {}
    if "since" in meta:
        out["x-version-introduced"] = meta["since"]
    if "deprecated_in" in meta:
        out["x-deprecated-in"] = meta["deprecated_in"]
    return out or None


# --------------------------- examples (safe synthesis) ---------------------------


def _placeholder(py_type: Any) -> Any:
    """Best-effort placeholders for builtins; nested models become {} or [] to avoid heavy imports."""
    origin = getattr(py_type, "__origin__", None)
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if py_type in (str,):
        return "example"
    if py_type in (int,):
        return 0
    if py_type in (float,):
        return 0.0
    if py_type in (bool,):
        return False
    # fallback
    return {}


def _example_for_pydantic(model_cls: type[object]) -> dict[str, Any]:
    """Synthesize a minimal example dict from model_fields (Pydantic v2)."""
    try:
        fields = getattr(model_cls, "model_fields", {})
    except Exception:
        return {}
    example: dict[str, Any] = {}
    for name, finfo in (fields or {}).items():
        # prefer default; else type-based placeholder
        if getattr(finfo, "default", None) is not None:
            example[name] = finfo.default
        else:
            ann = getattr(finfo, "annotation", None)
            example[name] = _placeholder(ann)
    return example


def _example_for_pandera(model_cls: type[object]) -> dict[str, Any]:
    """Produce a minimal 'row' example based on class attributes / schema JSON."""
    try:
        schema_json = model_cls.to_schema().to_json()
        data = json.loads(schema_json)
    except Exception:
        return {}
    # Pandera JSON often exposes columns under 'fields' or 'properties' after conversion.
    cols = []
    for key in ("fields", "properties", "columns"):
        obj = data.get(key)
        if isinstance(obj, dict):
            cols = list(obj.keys())
            break
    # One-row placeholder
    return dict.fromkeys(cols[:10], "example")


# --------------------------- schema transforms ---------------------------


def _apply_headers(schema: dict, module_name: str, class_name: str, base_url: str) -> None:
    """Apply headers.

    Parameters
    ----------
    schema : dict
        Description.
    module_name : str
        Description.
    class_name : str
        Description.
    base_url : str
        Description.

    Returns
    -------
    None
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _apply_headers(...)
    """
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["$id"] = f"{base_url.rstrip('/')}/{module_name}.{class_name}.json#"


def _inject_examples(schema: dict, example: dict | None) -> None:
    """Inject examples.

    Parameters
    ----------
    schema : dict
        Description.
    example : dict | None
        Description.

    Returns
    -------
    None
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _inject_examples(...)
    """
    if example:
        ex = schema.get("examples") or []
        if not isinstance(ex, list):
            ex = []
        # Only push if not present
        if example not in ex:
            ex = [example] + ex
        schema["examples"] = ex


def _inject_versions(schema: dict, versions: dict | None) -> None:
    """Inject versions.

    Parameters
    ----------
    schema : dict
        Description.
    versions : dict | None
        Description.

    Returns
    -------
    None
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _inject_versions(...)
    """
    if versions:
        # prefer root-level x- fields so they survive tools that merge properties
        for k, v in versions.items():
            schema[k] = v


# --------------------------- drift summarizer ---------------------------


def _diff_summary(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Summarize differences: top-level keys +/-; property changes (top 10)."""
    out: dict[str, Any] = {}
    old_keys, new_keys = set(old.keys()), set(new.keys())
    add_keys = sorted(list(new_keys - old_keys))
    del_keys = sorted(list(old_keys - new_keys))
    out["top_level_added"] = add_keys
    out["top_level_removed"] = del_keys

    def prop_keys(d: dict) -> set[str]:
        """
        Compute prop keys.
        
        Carry out the prop keys operation.
        
        Parameters
        ----------
        d : Mapping[str, Any]
            Description for ``d``.
        
        Returns
        -------
        Set[str]
            Description of return value.
        """
        props = d.get("properties")
        return set(props.keys()) if isinstance(props, dict) else set()

    pk_old, pk_new = prop_keys(old), prop_keys(new)
    prop_added = sorted(list(pk_new - pk_old))[:10]
    prop_removed = sorted(list(pk_old - pk_new))[:10]
    out["properties_added_top10"] = prop_added
    out["properties_removed_top10"] = prop_removed
    return out


# --------------------------- exporter core ---------------------------


@dataclass
class Cfg:
    """Represent Cfg.

    Attributes
    ----------
    attribute : type
        Description.

    Methods
    -------
    method()
        Description.

    Examples
    --------
    >>> Cfg(...)
    """

    ref_template: str
    base_url: str
    by_alias: bool
    check_drift: bool


def _export_one_pydantic(
    module_name: str, name: str, model_cls: type[object], cfg: Cfg, nav: dict
) -> tuple[Path, dict]:
    """Export one pydantic.

    Parameters
    ----------
    module_name : str
        Description.
    name : str
        Description.
    model_cls
        Description.
    cfg : Cfg
        Description.
    nav : dict
        Description.

    Returns
    -------
    tuple[Path, dict]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _export_one_pydantic(...)
    """
    # Generate schema dict (JSON schema 2020-12). Pydantic v2 exposes ref_template.
    # Docs: model_json_schema and ref_template customization.
    # https://docs.pydantic.dev/usage/schema/
    kwargs = {}
    if cfg.by_alias:
        kwargs["by_alias"] = True
    if cfg.ref_template:
        kwargs["ref_template"] = cfg.ref_template

    schema = model_cls.model_json_schema(**kwargs)  # dict
    _apply_headers(schema, module_name, name, cfg.base_url)
    _inject_versions(schema, _nav_versions(module_name, name, nav))
    _inject_examples(schema, _example_for_pydantic(model_cls))
    return (OUT / f"{module_name}.{name}.json", _sorted(schema))


def _export_one_pandera(
    module_name: str, name: str, model_cls: type[object], cfg: Cfg, nav: dict
) -> tuple[Path, dict]:
    """Export one pandera.

    Parameters
    ----------
    module_name : str
        Description.
    name : str
        Description.
    model_cls
        Description.
    cfg : Cfg
        Description.
    nav : dict
        Description.

    Returns
    -------
    tuple[Path, dict]
        Description.

    Raises
    ------
    Exception
        Description.

    Examples
    --------
    >>> _export_one_pandera(...)
    """
    # Pandera emits JSON string; normalize to dict, enrich, and sort.
    try:
        schema = json.loads(model_cls.to_schema().to_json())
    except Exception:
        return (OUT / f"{module_name}.{name}.json", {})
    schema = dict(schema)
    _apply_headers(schema, module_name, name, cfg.base_url)
    _inject_versions(schema, _nav_versions(module_name, name, nav))
    _inject_examples(schema, _example_for_pandera(model_cls))
    return (OUT / f"{module_name}.{name}.json", _sorted(schema))


def _iter_models() -> Iterator[tuple[str, str, object]]:
    """Yield Pydantic and Pandera models discovered under ``TOP_PACKAGES``.

    Returns
    -------
    Iterator[tuple[str, str, object]]
        Description.

    Examples
    --------
    >>> _iter_models()
    """
    for module_name in _module_iter():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if is_pydantic_model(obj) or is_pandera_model(obj):
                yield module_name, name, obj


def main(argv: list[str] | None = None) -> int:
    """
    Run the export-schemas CLI.
    
    Carry out the main operation.
    
    Parameters
    ----------
    argv : List[str] | None
        Description for ``argv``.
    
    Returns
    -------
    int
        Description of return value.
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--ref-template",
        default="#/$defs/{model}",
        help="Pydantic $ref template (default Pydantic; examples: '#/$defs/{model}', '#/components/schemas/{model}')",
    )
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for $id (default: env SCHEMA_BASE_URL or https://kgfoundry/schemas)",
    )
    p.add_argument(
        "--by-alias", action="store_true", help="Generate Pydantic schemas using field aliases"
    )
    p.add_argument(
        "--check-drift",
        action="store_true",
        help="Do not write; fail with summary if any drift exists",
    )
    args = p.parse_args(argv or [])
    cfg = Cfg(
        ref_template=args.ref_template,
        base_url=args.base_url,
        by_alias=args.by_alias,
        check_drift=args.check_drift,
    )

    nav = _load_navmap()
    drift_summaries: dict[str, Any] = {}
    changed = False

    for module_name, name, obj in _iter_models():
        path: Path
        data: dict
        if is_pydantic_model(obj):
            path, data = _export_one_pydantic(module_name, name, obj, cfg, nav)
        else:
            path, data = _export_one_pandera(module_name, name, obj, cfg, nav)
        if not data:
            continue

        # Drift handling
        old = None
        if path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                old = None

        if cfg.check_drift:
            if old is None or _sorted(old) != data:
                drift_summaries[str(path)] = _diff_summary(old or {}, data)
                changed = True
            continue
        wrote, old_text, new_text = _write_if_changed(path, data)
        if wrote:
            drift_summaries[str(path)] = _diff_summary(
                json.loads(old_text) if old_text else {}, data
            )
            changed = True

    if drift_summaries:
        DRIFT_OUT.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_OUT.write_text(json.dumps(drift_summaries, indent=2) + "\n", encoding="utf-8")

    if cfg.check_drift and changed:
        print("[schemas] drift detected; see docs/_build/schema_drift.json")
        return 2

    print("[schemas] export complete; drift:", "yes" if changed else "no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
