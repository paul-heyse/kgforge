"""Runtime compatibility patches for development tooling.

Python automatically imports ``sitecustomize`` (when present on the import path)
after the standard site initialisation phase.  We use this hook to backfill
APIs that some third-party tooling expects but which are missing from the
current dependency build.  Keeping the shim isolated here prevents the rest of
the application code from depending on the workaround.
"""

from __future__ import annotations

from typing import Any, cast

try:
    # pydoclint depends on ``docstring_parser.common.DocstringYields`` which
    # is not available in every release of docstring-parser or its forks.
    from docstring_parser import common as _doc_common
except Exception:  # pragma: no cover - best effort compatibility shim
    _doc_common = None
else:
    if (
        _doc_common is not None
        and not hasattr(_doc_common, "DocstringYields")
        and hasattr(_doc_common, "DocstringReturns")
    ):

        class DocstringYields(_doc_common.DocstringReturns):
            """Compatibility stub for missing ``DocstringYields``."""

            def __init__(
                self,
                args: list[str],
                description: str | None,
                type_name: str | None,
                return_name: str | None = None,
            ) -> None:
                """Init  .

                Parameters
                ----------
                args : list[str]
                    Description.
                description : Optional[str]
                    Description.
                type_name : Optional[str]
                    Description.
                return_name : Optional[str]
                    Description.

                Returns
                -------
                None
                    Description.

                Raises
                ------
                Exception
                    Description.

                Examples
                --------
                >>> __init__(...)
                """
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
        _doc_cls = _doc_common.Docstring

        DocstringType = Any  # runtime only typing used for annotations

        if not hasattr(_doc_cls, "attrs"):

            def _docstring_attrs(self: DocstringType) -> list[object]:
                """Return documented attributes if parsing supports them."""
                return [meta for meta in self.meta if getattr(meta, "args", None) == ["attr"]]

            _doc_cls.attrs = property(_docstring_attrs)  # type: ignore[attr-defined]

        _yield_cls = getattr(_doc_common, "DocstringYields", None)
        if _yield_cls is None and hasattr(_doc_common, "DocstringReturns"):
            _yield_cls = _doc_common.DocstringReturns

        if _yield_cls is not None and not hasattr(_doc_cls, "yields"):

            def _docstring_yields(self: DocstringType) -> object | None:
                """Return the first yields section if available."""
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return None
                for meta in self.meta:
                    if isinstance(meta, _yield_cls):
                        return meta
                return None

            _doc_cls.yields = property(_docstring_yields)  # type: ignore[attr-defined]

        if _yield_cls is not None and not hasattr(_doc_cls, "many_yields"):

            def _docstring_many_yields(self: DocstringType) -> list[object]:
                """Return all yields sections."""
                if _yield_cls is None:  # pragma: no cover - defensive guard
                    return []
                return [meta for meta in self.meta if isinstance(meta, _yield_cls)]

            _doc_cls.many_yields = property(_docstring_many_yields)  # type: ignore[attr-defined]

        if not hasattr(_doc_cls, "size"):

            def _docstring_size(self: DocstringType) -> int:
                """Estimate docstring size for style heuristics."""
                parts: list[str] = []
                if self.short_description:
                    parts.append(self.short_description)
                if self.long_description:
                    parts.append(self.long_description)
                parts.extend(getattr(meta, "description", "") or "" for meta in self.meta)
                return sum(len(part) for part in parts)

            _doc_cls.size = property(_docstring_size)  # type: ignore[attr-defined]
