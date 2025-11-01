from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeVar

class CSTNode: ...

class CSTVisitor:
    def visit(self, node: CSTNode) -> bool: ...
    def leave(self, original_node: CSTNode, updated_node: CSTNode) -> CSTNode: ...

class CSTTransformer(CSTVisitor): ...
class BaseExpression(CSTNode): ...
class BaseStatement(CSTNode): ...

class Name(BaseExpression):
    value: str

    def __init__(self, value: str) -> None: ...

class Expr(BaseStatement):
    value: BaseExpression

    def __init__(self, value: BaseExpression) -> None: ...

class SimpleString(BaseExpression):
    value: str

    def __init__(self, value: str) -> None: ...

class SimpleStatementLine(BaseStatement):
    body: tuple[Expr, ...]

    def __init__(self, *, body: Sequence[BaseStatement]) -> None: ...

class Suite(CSTNode):
    body: tuple[BaseStatement, ...]

    def __init__(self, *, body: Sequence[BaseStatement]) -> None: ...
    def __iter__(self) -> Iterator[BaseStatement]: ...

_SelfFn = TypeVar("_SelfFn", bound="FunctionDef")
_SelfCls = TypeVar("_SelfCls", bound="ClassDef")
_SelfWith = TypeVar("_SelfWith", bound="With")
_SelfExceptHandler = TypeVar("_SelfExceptHandler", bound="ExceptHandler")

class FunctionDef(CSTNode):
    name: Name
    body: Suite

    def __init__(self, *, name: Name, body: Suite) -> None: ...
    def with_changes(self: _SelfFn, *, body: Suite | None = ...) -> _SelfFn: ...

class ClassDef(CSTNode):
    name: Name
    body: Suite

    def __init__(self, *, name: Name, body: Suite) -> None: ...
    def with_changes(self: _SelfCls, *, body: Suite | None = ...) -> _SelfCls: ...

class Module(CSTNode):
    body: Suite

    def __init__(self, *, body: Sequence[BaseStatement]) -> None: ...
    @property
    def code(self) -> str: ...
    def visit(self, transformer: CSTTransformer) -> Module: ...
    def with_changes(self, *, body: Suite | Sequence[BaseStatement] | None = ...) -> Module: ...

def parse_module(source: str) -> Module: ...

class Attribute(BaseExpression):
    value: BaseExpression
    attr: Name

    def __init__(self, *, value: BaseExpression, attr: Name) -> None: ...

class Call(BaseExpression):
    func: BaseExpression
    args: tuple[Arg, ...]

    def __init__(self, *, func: BaseExpression, args: Sequence[Arg] = ...) -> None: ...

class Arg(CSTNode):
    value: BaseExpression
    keyword: Name | None

    def __init__(self, value: BaseExpression, *, keyword: Name | None = None) -> None: ...

class BinaryOperation(BaseExpression):
    left: BaseExpression
    operator: BinaryOperator
    right: BaseExpression

    def __init__(
        self, *, left: BaseExpression, operator: BinaryOperator, right: BaseExpression
    ) -> None: ...

class BinaryOperator(CSTNode): ...

class Divide(BinaryOperator):
    def __init__(self) -> None: ...

class Import(BaseStatement):
    names: Sequence[ImportAlias]

    def __init__(self, *, names: Sequence[ImportAlias]) -> None: ...

class ImportFrom(BaseStatement):
    module: Name | None
    names: Sequence[ImportAlias]

    def __init__(
        self, *, module: Name | None = ..., names: Sequence[ImportAlias] = ...
    ) -> None: ...

class ImportAlias(CSTNode):
    name: Name

    def __init__(self, *, name: Name) -> None: ...

class With(BaseStatement):
    items: tuple[WithItem, ...]

    def __init__(self, *, items: Sequence[WithItem]) -> None: ...
    def with_changes(self: _SelfWith, *, items: tuple[WithItem, ...] | None = ...) -> _SelfWith: ...

class WithItem(CSTNode):
    item: BaseExpression
    asname: AsName | None

    def __init__(self, *, item: BaseExpression, asname: AsName | None = None) -> None: ...

class IndentedBlock(CSTNode):
    body: tuple[BaseStatement, ...]

    def __init__(self, *, body: Sequence[BaseStatement]) -> None: ...

class ExceptHandler(CSTNode):
    type: BaseExpression | None
    name: AsName | None
    body: IndentedBlock
    whitespace_after_except: SimpleWhitespace

    def __init__(
        self,
        *,
        body: IndentedBlock,
        type: BaseExpression | None = ...,  # noqa: A002
        name: AsName | None = ...,
        whitespace_after_except: SimpleWhitespace = ...,
    ) -> None: ...
    def with_changes(
        self: _SelfExceptHandler,
        *,
        body: IndentedBlock | None = ...,
        name: AsName | None = ...,
        type: BaseExpression | None = ...,  # noqa: A002
        whitespace_after_except: SimpleWhitespace | None = ...,
    ) -> _SelfExceptHandler: ...

class AsName(CSTNode):
    name: Name

    def __init__(self, *, name: Name) -> None: ...

class SimpleWhitespace(CSTNode):
    value: str

    def __init__(self, value: str) -> None: ...

class ParserSyntaxError(Exception): ...

__all__: Sequence[str]
