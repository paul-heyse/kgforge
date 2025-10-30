from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from types import ModuleType, TracebackType
from typing import overload
from warnings import WarningMessage

class CaptureResult[T]:
    out: T
    err: T

class CaptureFixture[T]:
    def readouterr(self) -> CaptureResult[T]: ...

class MonkeyPatch:
    @overload
    def setattr(
        self, target: object, name: str, value: object, *, raising: bool = True
    ) -> None: ...
    @overload
    def setattr(self, target: str, value: object, *, raising: bool = True) -> None: ...

    def setenv(self, name: str, value: str, prepend: str | None = ...) -> None: ...
    def delenv(self, name: str, raising: bool = ...) -> None: ...


class LogCaptureFixture:
    def clear(self) -> None: ...
    def at_level(self, level: int | str, logger: str | None = ...) -> AbstractContextManager[None]: ...

    @property
    def records(self) -> list[object]: ...

    @property
    def text(self) -> str: ...


class WarningsRecorder(AbstractContextManager["WarningsRecorder"]):
    @property
    def list(self) -> list[object]: ...

    def __enter__(self) -> WarningsRecorder: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...

    def pop(self, category: type[BaseException] | int = ...) -> WarningMessage: ...

class ExceptionInfo[TExc: BaseException]:
    value: TExc

    def __enter__(self) -> ExceptionInfo[TExc]: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: BaseException | None,
    ) -> bool: ...

def raises[TExc: BaseException](
    expected_exception: type[TExc],
) -> AbstractContextManager[ExceptionInfo[TExc]]: ...
class UsageError(Exception):
    ...


class Mark:
    args: Sequence[object]
    kwargs: Mapping[str, object]


class MarkDecorator:
    def __call__(self, *args: object, **kwargs: object) -> object: ...
    def __getattr__(self, name: str) -> "MarkDecorator": ...


class Config:
    def addinivalue_line(self, name: str, line: str) -> None: ...


class Item:
    def get_closest_marker(self, name: str) -> Mark | None: ...
    def add_marker(self, marker: object) -> None: ...
def skip(reason: str = ..., *, allow_module_level: bool = ...) -> None: ...
@overload
def fixture[T](func: Callable[..., T]) -> Callable[..., T]: ...

@overload
def fixture[T](
    *,
    name: str | None = None,
    scope: str | None = None,
    params: Sequence[object] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]: ...


mark: MarkDecorator


def importorskip(
    name: str,
    *,
    minversion: str | None = ...,
    reason: str | None = ...,
) -> ModuleType: ...
