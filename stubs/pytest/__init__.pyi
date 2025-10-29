from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import overload

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
def skip(reason: str) -> None: ...
def fixture[T](*, name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]: ...
