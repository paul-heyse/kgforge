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
    from docstring_parser.common import Docstring, DocstringAttr, DocstringReturns, DocstringYields

    DocstringType = Docstring
    DocstringReturnsType = DocstringReturns
    DocstringYieldsType = DocstringYields
    DocstringAttrType = DocstringAttr
else:  # pragma: no cover - typing helpers
    DocstringType = Any
    DocstringReturnsType = Any
    DocstringYieldsType = Any
    DocstringAttrType = Any


class _DocstringCommonModule(Protocol):
    """Describe  DocstringCommonModule.

    <!-- auto:docstring-builder v1 -->

    how instances collaborate with the surrounding package. Highlight
    how the class supports nearby modules to guide readers through the
    codebase.

    Parameters
    ----------
    *args : inspect._empty
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.






    Returns
    -------
    inspect._empty
        Describe return value.
    """

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
            """Describe DocstringYields.

            <!-- auto:docstring-builder v1 -->

            Describe the data structure and how instances collaborate with the surrounding package.
            Highlight how the class supports nearby modules to guide readers through the codebase.
            """

            def __init__(
                self,
                args: list[str],
                description: str | None,
                type_name: str | None,
                return_name: str | None = None,
            ) -> None:
                """Describe   init  .

                <!-- auto:docstring-builder v1 -->

                Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

                Parameters
                ----------
                args : list
                    TODO: describe ``args``.
                description : str | None, optional
                    TODO: describe ``description``.
                type_name : str | None, optional
                    TODO: describe ``type_name``.
                return_name : str | None, optional, by default None
                    TODO: describe ``return_name``.
                """
                super().__init__(
                    args,
                    description,
                    type_name,
                    return_name=return_name,
                    is_generator=True,
                )
                # Match behaviour expected by downstream tooling that inspects
                # ``DocstringReturns`` instances for generator blocks.
                self.is_generator = True

        DocstringYields.__module__ = _doc_common.DocstringReturns.__module__
        _doc_common_typing = cast(Any, _doc_common)
        _doc_common_typing.DocstringYields = DocstringYields

    if _doc_common is not None and hasattr(_doc_common, "Docstring"):
        _doc_cls: type[DocstringType] = _doc_common.Docstring

        if not hasattr(_doc_cls, "attrs"):

            def _docstring_attrs(self: DocstringType) -> list[DocstringAttrType]:
                """Describe  docstring attrs.

                <!-- auto:docstring-builder v1 -->

                Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

                Returns
                -------
                list
                    TODO: describe return value.
                """
                return [meta for meta in self.meta if getattr(meta, "args", None) == ["attr"]]

            _doc_cls.attrs = property(_docstring_attrs)

        _yield_cls: type[DocstringReturnsType] | None
        if hasattr(_doc_common, "DocstringYields"):
            _yield_cls = _doc_common.DocstringYields
        elif hasattr(_doc_common, "DocstringReturns"):
            _yield_cls = _doc_common.DocstringReturns
        else:
            _yield_cls = None

        if _yield_cls is not None and not hasattr(_doc_cls, "yields"):

            def _docstring_yields(self: DocstringType) -> DocstringYieldsType | None:
                """Describe  docstring yields.

                <!-- auto:docstring-builder v1 -->

                Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

                Returns
                -------
                object | None
                    TODO: describe return value.
                """
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return None
                for meta in self.meta:
                    if isinstance(meta, _yield_cls):
                        return cast("DocstringYieldsType", meta)
                return None

            _doc_cls.yields = property(_docstring_yields)

        if _yield_cls is not None and not hasattr(_doc_cls, "many_yields"):

            def _docstring_many_yields(self: DocstringType) -> list[DocstringYieldsType]:
                """Describe  docstring many yields.

                <!-- auto:docstring-builder v1 -->

                Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

                Returns
                -------
                list
                    TODO: describe return value.
                """
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return []
                return [
                    cast("DocstringYieldsType", meta)
                    for meta in self.meta
                    if isinstance(meta, _yield_cls)
                ]

            _doc_cls.many_yields = property(_docstring_many_yields)

        if not hasattr(_doc_cls, "size"):

            def _docstring_size(self: DocstringType) -> int:
                """Describe  docstring size.

                <!-- auto:docstring-builder v1 -->

                Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

                Returns
                -------
                int
                    TODO: describe return value.
                """
                parts: list[str] = []
                if self.short_description:
                    parts.append(self.short_description)
                if self.long_description:
                    parts.append(self.long_description)
                parts.extend(getattr(meta, "description", "") or "" for meta in self.meta)
                return sum(len(part) for part in parts)

            _doc_cls.size = property(_docstring_size)
