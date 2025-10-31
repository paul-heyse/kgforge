from __future__ import annotations

from collections.abc import Sequence

class CSTNode: ...
class CSTVisitor: ...

class CSTTransformer:
    def __init__(self) -> None: ...

class CSTValidationError(Exception): ...
class CSTLogicError(Exception): ...
class ParserSyntaxError(Exception): ...
class BaseExpression(CSTNode): ...

class Name(BaseExpression):
    value: str

    def __init__(self, value: str) -> None: ...

class Attribute(BaseExpression):
    value: BaseExpression
    attr: Name

    def __init__(self, value: BaseExpression, attr: Name) -> None: ...
    def with_changes(
        self, *, value: BaseExpression | None = ..., attr: Name | None = ...
    ) -> Attribute: ...

class Arg(CSTNode):
    value: BaseExpression
    keyword: Name | None

    def __init__(self, value: BaseExpression, keyword: Name | None = ...) -> None: ...

class Divide(CSTNode): ...

class BinaryOperation(BaseExpression):
    left: BaseExpression
    operator: CSTNode
    right: BaseExpression

    def __init__(self, left: BaseExpression, operator: CSTNode, right: BaseExpression) -> None: ...

class Call(BaseExpression):
    func: BaseExpression
    args: tuple[Arg, ...]

    def __init__(self, *, func: BaseExpression, args: Sequence[Arg] | None = ...) -> None: ...
    def with_changes(
        self, *, func: BaseExpression | None = ..., args: Sequence[Arg] | None = ...
    ) -> Call: ...

class AsName(CSTNode):
    name: Name

    def __init__(self, name: Name) -> None: ...

class IndentedBlock(CSTNode):
    body: tuple[CSTNode, ...]

    def __init__(self, *, body: Sequence[CSTNode]) -> None: ...

class SimpleWhitespace(CSTNode):
    value: str

    def __init__(self, value: str) -> None: ...

class ExceptHandler(CSTNode):
    type: BaseExpression | None
    name: AsName | None
    body: IndentedBlock
    whitespace_after_except: SimpleWhitespace

    def with_changes(
        self,
        *,
        type: BaseExpression | None = ...,
        name: AsName | None = ...,
        body: IndentedBlock | None = ...,
        whitespace_after_except: SimpleWhitespace | None = ...,
    ) -> ExceptHandler: ...

class WithItem(CSTNode):
    item: BaseExpression
    asname: AsName | None

    def __init__(self, item: BaseExpression, asname: AsName | None = ...) -> None: ...

class With(CSTNode):
    items: tuple[WithItem, ...]

    def with_changes(self, *, items: Sequence[WithItem] | None = ...) -> With: ...

class ImportAlias(CSTNode):
    name: Name
    asname: AsName | None

    def __init__(self, name: Name, asname: AsName | None = ...) -> None: ...

class Import(CSTNode):
    names: tuple[ImportAlias, ...]

    def __init__(self, names: Sequence[ImportAlias]) -> None: ...

class ImportFrom(CSTNode):
    module: Name | None
    names: Sequence[ImportAlias]

    def __init__(self, module: Name | None, names: Sequence[ImportAlias]) -> None: ...

class SimpleStatementLine(CSTNode):
    body: tuple[CSTNode, ...]

    def __init__(self, body: Sequence[CSTNode]) -> None: ...

class Module(CSTNode):
    body: tuple[CSTNode, ...]

    @property
    def code(self) -> str: ...
    def with_changes(self, *, body: Sequence[CSTNode] | None = ...) -> Module: ...
    def visit(self, transformer: CSTTransformer) -> Module: ...

def parse_module(source: str) -> Module: ...

__all__: Sequence[str]
