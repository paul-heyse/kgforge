#!/usr/bin/env python3
"""Overview of export schemas.

This module bundles export schemas logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pkgutil
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from tools._shared.logging import get_logger

if TYPE_CHECKING:
    pass

LOGGER = get_logger(__name__)

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


def _load_pandera_module() -> ModuleType | None:
    """Load ``pandera`` using ``importlib`` to avoid strict import-time failures."""
    try:
        module = importlib.import_module("pandera")
    except ModuleNotFoundError:
        return None
    return module


# --------------------------- utils ---------------------------


def _sorted(obj: object) -> object:
    """Recursively sort dict keys; leave arrays as-is for semantics."""
    if isinstance(obj, dict):
        return {k: _sorted(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sorted(x) for x in obj]
    try:
        core = importlib.import_module("pydantic_core")
    except ModuleNotFoundError:
        pass
    else:
        undefined_attr = "PydanticUndefined"
        if getattr(core, undefined_attr, None) is obj:
            return None
    return obj


def _write_if_changed(
    path: Path, data: Mapping[str, object]
) -> tuple[bool, str | None, str | None]:
    """Write JSON with sorted keys + trailing newline if content changed.

    Return (changed, old, new).
    """
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
        except (ImportError, ModuleNotFoundError) as exc:
            LOGGER.debug("Failed to import package %s: %s", pkg, exc)
            continue
        if not hasattr(module, "__path__"):
            continue
        for info in pkgutil.walk_packages(module.__path__, prefix=f"{pkg}."):
            yield info.name


def is_pydantic_model(obj: object) -> bool:
    """Compute is pydantic model.

    Carry out the is pydantic model operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    obj : object
        Description for ``obj``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.docs.export_schemas import is_pydantic_model
    >>> result = is_pydantic_model(...)
    >>> result  # doctest: +ELLIPSIS
    """
    base_model = _load_pydantic_base_model()
    if base_model is None:
        return False
    return inspect.isclass(obj) and issubclass(obj, base_model) and obj is not base_model


def is_pandera_model(obj: object) -> bool:
    """Compute is pandera model.

    Carry out the is pandera model operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    obj : object
        Description for ``obj``.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> from tools.docs.export_schemas import is_pandera_model
    >>> result = is_pandera_model(...)
    >>> result  # doctest: +ELLIPSIS
    """
    pa = _load_pandera_module()
    if pa is None:
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
        return cast(dict[str, Any], json.loads(NAVMAP.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


def _load_pydantic_base_model() -> type[Any] | None:
    try:
        module = importlib.import_module("pydantic")
    except ImportError:
        return None
    base_model = getattr(module, "BaseModel", None)
    if not isinstance(base_model, type):
        return None
    return base_model


def _nav_versions(module_name: str, class_name: str, nav: dict[str, Any]) -> dict[str, Any] | None:
    """Find navmap version metadata.

    Return {x-version-introduced, x-deprecated-in} when module and symbol names match.
    """
    mods = nav.get("modules") or {}
    entry = mods.get(module_name)
    if not entry:
        return None
    meta_map = entry.get("meta") or {}
    meta = meta_map.get(class_name) or meta_map.get(f"{module_name}.{class_name}") or {}
    out = {}
    if "since" in meta:
        out["x-version-introduced"] = meta["since"]
    if "deprecated_in" in meta:
        out["x-deprecated-in"] = meta["deprecated_in"]
    return out or None


# --------------------------- examples (safe synthesis) ---------------------------


def _placeholder(py_type: object) -> object:
    """Generate lightweight placeholders for basic Python types."""
    origin = getattr(py_type, "__origin__", None)
    if origin in {list, set, tuple}:
        return []
    if origin is dict:
        return {}

    builtin_defaults: dict[type[object], object] = {
        str: "example",
        int: 0,
        float: 0.0,
        bool: False,
    }
    if isinstance(py_type, type) and py_type in builtin_defaults:
        return builtin_defaults[py_type]
    return {}


def _example_for_pydantic(model_cls: type[object]) -> dict[str, object]:
    """Synthesize a minimal example dict from model_fields (Pydantic v2)."""
    try:
        fields = getattr(model_cls, "model_fields", {})
    except AttributeError:
        return {}
    example: dict[str, object] = {}
    for name, finfo in (fields or {}).items():
        # prefer default; else type-based placeholder
        if getattr(finfo, "default", None) is not None:
            example[name] = finfo.default
        else:
            ann = getattr(finfo, "annotation", None)
            example[name] = _placeholder(ann)
    return example


def _example_for_pandera(model_cls: type[object]) -> dict[str, object]:
    """Produce a minimal 'row' example based on class attributes / schema JSON."""
    attr_name = "to_schema"
    try:
        to_schema = getattr(model_cls, attr_name)
    except AttributeError:
        return {}

    try:
        schema_json = to_schema().to_json()
        data = json.loads(schema_json)
    except (AttributeError, json.JSONDecodeError, OSError):
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


def _apply_headers(
    schema: dict[str, object], module_name: str, class_name: str, base_url: str
) -> None:
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


def _inject_examples(schema: dict[str, object], example: Mapping[str, object] | None) -> None:
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
            ex = [example, *ex]
        schema["examples"] = ex


def _inject_versions(schema: dict[str, object], versions: Mapping[str, object] | None) -> None:
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


def _diff_summary(old: Mapping[str, object], new: Mapping[str, object]) -> dict[str, object]:
    """Summarize differences: top-level keys +/-; property changes (top 10)."""
    out: dict[str, object] = {}
    old_keys, new_keys = set(old.keys()), set(new.keys())
    add_keys = sorted(new_keys - old_keys)
    del_keys = sorted(old_keys - new_keys)
    out["top_level_added"] = add_keys
    out["top_level_removed"] = del_keys

    def prop_keys(d: Mapping[str, object]) -> set[str]:
        """Compute prop keys.

        Carry out the prop keys operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

        Parameters
        ----------
        d : collections.abc.Mapping
            Description for ``d``.

        Returns
        -------
        collections.abc.Set
            Description of return value.

        Examples
        --------
        >>> from tools.docs.export_schemas import prop_keys
        >>> result = prop_keys(...)
        >>> result  # doctest: +ELLIPSIS
        """
        props = d.get("properties")
        return set(props.keys()) if isinstance(props, dict) else set()

    pk_old, pk_new = prop_keys(old), prop_keys(new)
    prop_added = sorted(pk_new - pk_old)[:10]
    prop_removed = sorted(pk_old - pk_new)[:10]
    out["properties_added_top10"] = prop_added
    out["properties_removed_top10"] = prop_removed
    return out


# --------------------------- exporter core ---------------------------


@dataclass
class Cfg:
    """Model the Cfg.

    Represent the cfg data structure used throughout the project. The class encapsulates behaviour
    behind a well-defined interface for collaborating components. Instances are typically created by
    factories or runtime orchestrators documented nearby.
    """

    ref_template: str
    base_url: str
    by_alias: bool
    check_drift: bool


def _export_one_pydantic(
    module_name: str, name: str, model_cls: type[object], cfg: Cfg, nav: dict[str, Any]
) -> tuple[Path, dict[str, object]]:
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
    kwargs: dict[str, Any] = {}
    if cfg.by_alias:
        kwargs["by_alias"] = True
    if cfg.ref_template:
        kwargs["ref_template"] = cfg.ref_template

    model_any = cast(Any, model_cls)
    schema = cast(dict[str, object], model_any.model_json_schema(**kwargs))
    _apply_headers(schema, module_name, name, cfg.base_url)
    _inject_versions(schema, _nav_versions(module_name, name, nav))
    _inject_examples(schema, _example_for_pydantic(model_cls))
    return (OUT / f"{module_name}.{name}.json", cast(dict[str, object], _sorted(schema)))


def _export_one_pandera(
    module_name: str, name: str, model_cls: type[object], cfg: Cfg, nav: dict[str, Any]
) -> tuple[Path, dict[str, object]]:
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
    attr_name = "to_schema"
    try:
        to_schema = getattr(model_cls, attr_name)
    except AttributeError:
        return (OUT / f"{module_name}.{name}.json", {})

    try:
        schema_json = to_schema().to_json()
        schema_obj = json.loads(schema_json)
    except (AttributeError, json.JSONDecodeError, OSError):
        return (OUT / f"{module_name}.{name}.json", {})
    schema: dict[str, object] = dict(schema_obj)
    _apply_headers(schema, module_name, name, cfg.base_url)
    _inject_versions(schema, _nav_versions(module_name, name, nav))
    _inject_examples(schema, _example_for_pandera(model_cls))
    return (OUT / f"{module_name}.{name}.json", cast(dict[str, object], _sorted(schema)))


def _iter_models() -> Iterator[tuple[str, str, type[object]]]:
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
        except (ImportError, ModuleNotFoundError) as exc:
            LOGGER.debug("Failed to import module %s: %s", module_name, exc)
            continue
        for name, obj in vars(module).items():
            if is_pydantic_model(obj) or is_pandera_model(obj):
                model_cls = cast(type[object], obj)
                yield module_name, name, model_cls


def _parse_args(argv: Sequence[str] | None) -> Cfg:
    """Parse CLI arguments into a :class:`Cfg`."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-template",
        default="#/$defs/{model}",
        help="Pydantic $ref template (default Pydantic; examples: '#/$defs/{model}', '#/components/schemas/{model}')",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for $id (default: env SCHEMA_BASE_URL or https://kgfoundry/schemas)",
    )
    parser.add_argument(
        "--by-alias", action="store_true", help="Generate Pydantic schemas using field aliases"
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Do not write; fail with summary if any drift exists",
    )
    args = parser.parse_args(argv or [])
    return Cfg(
        ref_template=args.ref_template,
        base_url=args.base_url,
        by_alias=args.by_alias,
        check_drift=args.check_drift,
    )


def _load_existing_schema(path: Path) -> dict[str, object] | None:
    """Read an existing JSON schema from disk."""
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _schemas_match(previous: Mapping[str, object] | None, current: Mapping[str, object]) -> bool:
    """Return ``True`` when two schema mappings represent the same data."""
    if previous is None:
        return False
    return _sorted(previous) == current


def _record_drift(
    path: Path,
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    drift_summaries: dict[str, object],
) -> None:
    """Store a drift summary for ``path`` in ``drift_summaries``."""
    drift_summaries[str(path)] = _diff_summary(previous or {}, current)


def _export_model(
    module_name: str,
    name: str,
    model_cls: type[object],
    cfg: Cfg,
    nav: dict[str, Any],
) -> tuple[Path, dict[str, object]]:
    """Dispatch to the correct exporter based on model type."""
    if is_pydantic_model(model_cls):
        return _export_one_pydantic(module_name, name, model_cls, cfg, nav)
    return _export_one_pandera(module_name, name, model_cls, cfg, nav)


def _remove_stale_schemas(
    produced_paths: set[Path], cfg: Cfg, drift_summaries: dict[str, object]
) -> bool:
    """Remove schema files that were not produced in the current run."""
    changed = False
    existing_paths = set(OUT.glob("*.json"))
    for stale_path in sorted(existing_paths - produced_paths):
        old_data = _load_existing_schema(stale_path) or {}
        _record_drift(stale_path, old_data, {}, drift_summaries)
        changed = True
        if cfg.check_drift:
            continue
        with suppress(FileNotFoundError):
            stale_path.unlink()
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Compute main."""
    cfg = _parse_args(argv)
    nav = _load_navmap()
    drift_summaries: dict[str, object] = {}
    changed = False
    produced_paths: set[Path] = set()

    for module_name, name, model_cls in _iter_models():
        path, data = _export_model(module_name, name, model_cls, cfg, nav)
        if not data:
            continue
        produced_paths.add(path)

        previous = _load_existing_schema(path)
        if cfg.check_drift:
            if not _schemas_match(previous, data):
                _record_drift(path, previous, data, drift_summaries)
                changed = True
            continue

        wrote, _, _ = _write_if_changed(path, data)
        if wrote:
            _record_drift(path, previous or {}, data, drift_summaries)
            changed = True

    if _remove_stale_schemas(produced_paths, cfg, drift_summaries):
        changed = True

    if drift_summaries:
        DRIFT_OUT.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_OUT.write_text(json.dumps(drift_summaries, indent=2) + "\n", encoding="utf-8")

    if cfg.check_drift and changed:
        LOGGER.warning("[schemas] drift detected; see docs/_build/schema_drift.json")
        return 2

    LOGGER.info("[schemas] export complete; drift: %s", "yes" if changed else "no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
