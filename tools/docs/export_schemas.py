"""Export Schemas utilities."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path

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


def is_pydantic_model(obj: object) -> bool:
    """Compute is pydantic model.

    Carry out the is pydantic model operation.

    Parameters
    ----------
    obj : object
        Description for ``obj``.

    Returns
    -------
    bool
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    try:
        from pydantic import BaseModel
    except Exception:
        return False
    return inspect.isclass(obj) and issubclass(obj, BaseModel)


def is_pandera_model(obj: object) -> bool:
    """Compute is pandera model.

    Carry out the is pandera model operation.

    Parameters
    ----------
    obj : object
        Description for ``obj``.

    Returns
    -------
    bool
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    try:
        import pandera as pa
    except Exception:
        return False
    schema_model = getattr(pa, "SchemaModel", None)
    if schema_model is None:
        return False
    return inspect.isclass(obj) and issubclass(obj, schema_model)


def iter_packages() -> list[str]:
    """Compute iter packages.

    Carry out the iter packages operation.

    Returns
    -------
    List[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    discovered: list[str] = []
    for pkg in TOP_PACKAGES:
        try:
            module = importlib.import_module(pkg)
        except Exception:
            continue
        if not hasattr(module, "__path__"):
            continue
        for info in pkgutil.walk_packages(module.__path__, prefix=f"{pkg}."):
            discovered.append(info.name)
    return discovered


def export_schema(module_name: str, name: str, obj: object) -> None:
    """Compute export schema.

    Carry out the export schema operation.

    Parameters
    ----------
    module_name : str
        Description for ``module_name``.
    name : str
        Description for ``name``.
    obj : object
        Description for ``obj``.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    filename = f"{module_name}.{name}.json"
    path = OUT / filename
    if is_pydantic_model(obj):
        try:
            schema = obj.model_json_schema()
        except Exception:
            return
        path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    elif is_pandera_model(obj):
        try:
            schema = obj.to_schema().to_json()
        except Exception:
            return
        path.write_text(schema + ("" if schema.endswith("\n") else "\n"), encoding="utf-8")


def main() -> None:
    """Compute main.

    Carry out the main operation.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    exported = 0
    for module_name in iter_packages():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name, obj in vars(module).items():
            if is_pydantic_model(obj) or is_pandera_model(obj):
                export_schema(module_name, name, obj)
                exported += 1
    print(f"exported {exported} schemas to {OUT}")


if __name__ == "__main__":
    main()
