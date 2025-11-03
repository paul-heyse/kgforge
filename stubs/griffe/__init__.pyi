from typing import Any

class Object:
    path: str
    canonical_path: Object | None
    kind: Any
    members: dict[str, Object] | None
    docstring: Any
    relative_package_filepath: Any
    relative_filepath: Any
    lineno: int | float | None
    endlineno: int | float | None
    is_async: bool | None
    is_property: bool | None

class GriffeLoader:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def load(self, name: str) -> Any: ...
    def load_module(self, name: str) -> Any: ...

class Module(Object): ...
class Member(Object): ...
