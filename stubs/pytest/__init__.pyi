from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from re import Pattern
from types import ModuleType, TracebackType
from typing import Any, TypeVar, overload
from warnings import WarningMessage

T = TypeVar("T")
TW = TypeVar("TW", bound=BaseException)

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
    def set_level(self, level: int | str, logger: str | None = ...) -> None: ...
    def at_level(
        self, level: int | str, logger: str | None = ...
    ) -> AbstractContextManager[None]: ...
    @property
    def records(self) -> list[logging.LogRecord]: ...
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

@overload
def raises[TExc: BaseException](
    expected_exception: type[TExc],
    *,
    match: str | Pattern[str] | None = ...,
) -> AbstractContextManager[ExceptionInfo[TExc]]: ...
@overload
def raises[TExc: BaseException](
    expected_exception: tuple[type[TExc], ...],
    *,
    match: str | Pattern[str] | None = ...,
) -> AbstractContextManager[ExceptionInfo[TExc]]: ...

class UsageError(Exception): ...

class Mark:
    args: Sequence[object]
    kwargs: Mapping[str, object]

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

mark: Any

def importorskip(
    name: str,
    *,
    minversion: str | None = ...,
    reason: str | None = ...,
) -> ModuleType: ...
def warns(
    expected_warning: type[Warning] | tuple[type[Warning], ...],
    *,
    match: str | Pattern[str] | None = ...,
) -> AbstractContextManager[list[WarningMessage]]: ...

class _ApproxReturn:
    def __iter__(self) -> Iterator[float]: ...

@overload
def approx(
    expected: float, *, rel: float | None = ..., abs: float | None = ...
) -> _ApproxReturn: ...
@overload
def approx(
    expected: Sequence[float],
    *,
    rel: float | None = ...,
    abs: float | None = ...,
) -> Sequence[_ApproxReturn]: ...
