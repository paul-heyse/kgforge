from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "BaseStatement",
    "CSTNode",
    "CSTTransformer",
    "CSTVisitor",
    "ClassDef",
    "Expr",
    "FunctionDef",
    "Module",
    "Name",
    "SimpleStatementLine",
    "SimpleString",
    "parse_module",
]

class CSTNode: ...

class CSTVisitor:
    def visit(self, node: CSTNode) -> None: ...

class CSTTransformer(CSTVisitor):
    pass

class BaseStatement(CSTNode): ...

class Name(CSTNode):
    value: str

class Expr(CSTNode):
    value: CSTNode
    def __init__(self, value: CSTNode) -> None: ...

class SimpleString(CSTNode):
    value: str
    def __init__(self, value: str) -> None: ...

class SimpleStatementLine(BaseStatement):
    body: Sequence[CSTNode]
    def __init__(self, *, body: Sequence[CSTNode]) -> None: ...

class FunctionDef(BaseStatement):
    name: Name
    body: Any
    def with_changes(self, **changes: object) -> FunctionDef: ...

class ClassDef(BaseStatement):
    name: Name
    body: Any
    def with_changes(self, **changes: object) -> ClassDef: ...

class Module(CSTNode):
    body: Any
    code: str
    def visit(self, visitor: CSTVisitor) -> Module: ...
    def with_changes(self, **changes: object) -> Module: ...

def parse_module(code: str) -> Module: ...
