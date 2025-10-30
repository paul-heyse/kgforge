from __future__ import annotations

from collections.abc import Mapping

class Undefined:
    ...


class StrictUndefined(Undefined):
    ...


class Template:
    def render(self, *args: object, **kwargs: object) -> str: ...


class Environment:
    def __init__(
        self,
        *,
        undefined: type[Undefined] | None = None,
        trim_blocks: bool | None = None,
        lstrip_blocks: bool | None = None,
        globals: Mapping[str, object] | None = None,
    ) -> None: ...

    def from_string(self, source: str) -> Template: ...
