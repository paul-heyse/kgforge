from __future__ import annotations

from collections.abc import Sequence
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

class Expr(BaseStatement):
    value: BaseExpression

class SimpleString(BaseExpression):
    value: str

class SimpleStatementLine(BaseStatement):
    body: tuple[Expr, ...]

class Suite(CSTNode):
    body: tuple[BaseStatement, ...]

_SelfFn = TypeVar("_SelfFn", bound="FunctionDef")
_SelfCls = TypeVar("_SelfCls", bound="ClassDef")

class FunctionDef(CSTNode):
    name: Name
    body: Suite

    def with_changes(self: _SelfFn, *, body: Suite | None = ...) -> _SelfFn: ...

class ClassDef(CSTNode):
    name: Name
    body: Suite

    def with_changes(self: _SelfCls, *, body: Suite | None = ...) -> _SelfCls: ...

class Module(CSTNode):
    body: Suite

    @property
    def code(self) -> str: ...
    def visit(self, transformer: CSTTransformer) -> Module: ...

def parse_module(source: str) -> Module: ...

__all__: Sequence[str]
