"""Runtime compatibility patches for development tooling.

Python automatically imports ``sitecustomize`` (when present on the import path)
after the standard site initialisation phase.  We use this hook to backfill
APIs that some third-party tooling expects but which are missing from the
current dependency build.  Keeping the shim isolated here prevents the rest of
the application code from depending on the workaround.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from docstring_parser.common import Docstring
    from docstring_parser.common import DocstringReturns

    DocstringType = Docstring
    DocstringReturnsType = DocstringReturns
else:  # pragma: no cover - typing helpers
    DocstringType = Any
    DocstringReturnsType = Any


class _DocstringCommonModule(Protocol):
    Docstring: type[DocstringType]
    DocstringReturns: type[DocstringReturnsType]
    DocstringYields: type[DocstringReturnsType]

_doc_common: _DocstringCommonModule | None = None

try:
    # pydoclint depends on ``docstring_parser.common.DocstringYields`` which
    # is not available in every release of docstring-parser or its forks.
    from docstring_parser import common as _imported_doc_common
except Exception:  # pragma: no cover - best effort compatibility shim
    pass
else:
    _doc_common = cast(_DocstringCommonModule, _imported_doc_common)
    if (
        _doc_common is not None
        and not hasattr(_doc_common, "DocstringYields")
        and hasattr(_doc_common, "DocstringReturns")
    ):

        class DocstringYields(_doc_common.DocstringReturns):
            """Compatibility stub for missing ``DocstringYields``.

            Parameters
            ----------
            args : list[str]
                Names associated with the yielded values.
            description : str | None
                Optional description rendered in the docstring.
            type_name : str | None
                Type annotation captured for the yield expression.
            return_name : str | None, optional
                Explicit name attached to the yield block.
            """
        docstring_returns_class: type[DocstringReturnsType] = _doc_common.DocstringReturns

        def _docstring_yields_init(
            self: DocstringReturnsType,
            args: list[str],
            description: str | None,
            type_name: str | None,
            return_name: str | None = None,
        ) -> None:
            """Compatibility constructor for missing ``DocstringYields``."""

            docstring_returns_class.__init__(
                self,
                args: list[str],
                description: str | None,
                type_name: str | None,
                return_name: str | None = None,
            ) -> None:
                super().__init__(
                    args=args,
                    description=description,
                    type_name=type_name,
                    is_generator=True,
                    return_name=return_name,
                )

        DocstringYields.__module__ = _doc_common.DocstringReturns.__module__
        _doc_common_typing = cast(Any, _doc_common)
        _doc_common_typing.DocstringYields = DocstringYields

    if _doc_common is not None and hasattr(_doc_common, "Docstring"):
        _doc_cls: type[DocstringType] = _doc_common.Docstring

        if not hasattr(_doc_cls, "attrs"):

            def _docstring_attrs(self: DocstringType) -> list[object]:
                """Return documented attributes if parsing supports them.
"""
                return [meta for meta in self.meta if getattr(meta, "args", None) == ["attr"]]

            setattr(_doc_cls, "attrs", property(_docstring_attrs))

        _yield_cls: type[DocstringReturnsType] | None
        if hasattr(_doc_common, "DocstringYields"):
            _yield_cls = _doc_common.DocstringYields
        elif hasattr(_doc_common, "DocstringReturns"):
            _yield_cls = _doc_common.DocstringReturns
        else:
            _yield_cls = None

        if _yield_cls is not None and not hasattr(_doc_cls, "yields"):

            def _docstring_yields(self: DocstringType) -> object | None:
                """Return the first yields section if available.
"""
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return None
                for meta in self.meta:
                    if isinstance(meta, _yield_cls):
                        return meta
                return None

            setattr(_doc_cls, "yields", property(_docstring_yields))

        if _yield_cls is not None and not hasattr(_doc_cls, "many_yields"):

            def _docstring_many_yields(self: DocstringType) -> list[object]:
                """Return all yields sections.
"""
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return []
                return [meta for meta in self.meta if isinstance(meta, _yield_cls)]

            setattr(_doc_cls, "many_yields", property(_docstring_many_yields))

        if not hasattr(_doc_cls, "size"):

            def _docstring_size(self: DocstringType) -> int:
                """Estimate docstring size for style heuristics.
"""
                parts: list[str] = []
                if self.short_description:
                    parts.append(self.short_description)
                if self.long_description:
                    parts.append(self.long_description)
                parts.extend(getattr(meta, "description", "") or "" for meta in self.meta)
                return sum(len(part) for part in parts)

            setattr(_doc_cls, "size", property(_docstring_size))
