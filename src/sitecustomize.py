"""Typed compatibility shims for :mod:`docstring_parser`.

The docstring builder relies on attributes that are not available in every
release of :mod:`docstring_parser`.  This module patches the library at import
 time while keeping the mutations visible to static type checkers.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

logger = logging.getLogger("kgfoundry.docstring_shim")

_FLAG_VALUE = os.environ.get("KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE", "1").strip().lower()
ENABLE_SITECUSTOMIZE = _FLAG_VALUE not in {"0", "false", "off", "no"}

if TYPE_CHECKING:  # pragma: no cover - imported solely for typing information
    from docstring_parser.common import Docstring, DocstringParam, DocstringReturns

    class DocstringYields(DocstringReturns):
        """Type-checking placeholder for generator metadata entries."""

        ...
else:  # pragma: no cover - imported lazily when type checking is disabled
    Docstring = Any
    DocstringParam = Any
    DocstringReturns = Any
    DocstringYields = Any


class DocstringMetaProto(Protocol):
    """Common surface shared by docstring ``meta`` entries.

    ``docstring_parser`` represents return blocks, attributes, and parameter
    annotations as ``meta`` objects.  Each metadata entry exposes a short
    textual description that may be ``None`` when the source docstring omitted
    additional context.  The concrete subclasses supplied by
    ``docstring_parser`` add extra fields, but they all share this description
    surface which is the only part we need for compatibility.
    """

    description: str | None


class DocstringAttrProto(DocstringMetaProto, Protocol):
    """Protocol describing attribute metadata entries.

    ``docstring_parser`` represents both attributes and parameters using
    :class:`DocstringParam`.  Attribute metadata entries expose their marker and
    attribute name via ``args`` while reusing ``description`` for human readable
    explanations.  We only need the attribute name list to populate the
    ``attrs`` property in the shimmed ``Docstring`` objects.
    """

    args: Sequence[str]


class DocstringReturnsProto(DocstringMetaProto, Protocol):
    """Protocol describing return metadata entries."""

    args: Sequence[str]
    type_name: str | None
    is_generator: bool
    return_name: str | None

    def __init__(
        self,
        args: Sequence[str],
        description: str | None,
        type_name: str | None,
        *,
        is_generator: bool,
        return_name: str | None = None,
    ) -> None:
        """Initialise a return metadata entry."""


class DocstringYieldsProto(DocstringReturnsProto, Protocol):
    """Protocol describing generator metadata entries.

    ``docstring_parser`` treats ``Yields`` sections as a specialisation of
    ``Returns``.  Downstream tools expect a small set of attributes that mark a
    metadata block as representing a generator.  The protocol mirrors those
    attributes so we can annotate our compatibility shim precisely.
    """

    args: Sequence[str]
    type_name: str | None
    is_generator: bool
    return_name: str | None


class DocstringProto(Protocol):
    """Protocol describing the subset of :class:`Docstring` we extend.

    Only a handful of properties are accessed by the compatibility helpers:
    the metadata list and the short/long descriptions.  Limiting the protocol
    to those surfaces keeps mypy satisfied without reimplementing the entire
    third-party type.
    """

    meta: list[DocstringMetaProto]
    short_description: str | None
    long_description: str | None


class DocstringCommonModuleProto(Protocol):
    """Runtime module exported by :mod:`docstring_parser.common`.

    The real module exposes several concrete classes that the shim patches at runtime.  Describing
    the attribute names here allows callers to keep strong typing while still deferring to the
    imported module for the actual implementations.
    """

    Docstring: type[DocstringProto]
    DocstringAttr: type[DocstringAttrProto]
    DocstringParam: type[DocstringAttrProto]
    DocstringReturns: type[DocstringReturnsProto]
    DocstringYields: type[DocstringYieldsProto]


def ensure_docstring_attrs[DocT: DocstringProto, AttrT: DocstringAttrProto](
    doc_cls: type[DocT], attr_cls: type[AttrT]
) -> bool:
    """Install an ``attrs`` property on ``doc_cls`` when absent.

    Parameters
    ----------
    doc_cls : type[DocT]
        Docstring class to mutate in place.
    attr_cls : type[AttrT]
        Metadata type that identifies attribute entries.

    Returns
    -------
    bool
        ``True`` if the property was added, otherwise ``False``.
    """
    if hasattr(doc_cls, "attrs"):
        logger.debug("Docstring attrs already present on %%s", doc_cls)
        return False

    def _attrs(self: DocT) -> list[AttrT]:
        collected: list[AttrT] = []
        for entry in self.meta:
            if isinstance(entry, attr_cls):
                args = list(entry.args)
                if args and args[0].lower() == "attribute":
                    collected.append(entry)
        return collected

    doc_cls_any = cast(Any, doc_cls)
    doc_cls_any.attrs = property(_attrs)
    logger.debug("Registered attrs property on %%s", doc_cls)
    return True


def ensure_docstring_yields[DocT: DocstringProto, YieldsT: DocstringYieldsProto](
    doc_cls: type[DocT], yields_cls: type[YieldsT]
) -> tuple[bool, bool]:
    """Ensure docstring instances expose generator metadata helpers.

    Parameters
    ----------
    doc_cls : type[DocT]
        Docstring class to mutate in place.
    yields_cls : type[YieldsT]
        Metadata type representing ``Yields`` blocks.

    Returns
    -------
    tuple[bool, bool]
        Flags indicating whether ``yields`` and ``many_yields`` were installed.
    """
    added_single = False
    added_many = False

    if not hasattr(doc_cls, "yields"):

        def _yield(self: DocT) -> YieldsT | None:
            for entry in self.meta:
                if isinstance(entry, yields_cls):
                    return entry
            return None

        doc_cls_any = cast(Any, doc_cls)
        doc_cls_any.yields = property(_yield)
        added_single = True
        logger.debug("Registered yields property on %%s", doc_cls)

    if not hasattr(doc_cls, "many_yields"):

        def _yield_many(self: DocT) -> list[YieldsT]:
            matches: list[YieldsT] = []
            for entry in self.meta:
                if isinstance(entry, yields_cls):
                    matches.append(entry)
            return matches

        doc_cls_any = cast(Any, doc_cls)
        doc_cls_any.many_yields = property(_yield_many)
        added_many = True
        logger.debug("Registered many_yields property on %%s", doc_cls)

    return added_single, added_many


def ensure_docstring_size[DocT: DocstringProto](doc_cls: type[DocT]) -> bool:
    """Expose a ``size`` property summarising docstring content length.

    Parameters
    ----------
    doc_cls : type[DocT]
        Docstring class to mutate in place.

    Returns
    -------
    bool
        ``True`` if the property was added, otherwise ``False``.
    """
    if hasattr(doc_cls, "size"):
        logger.debug("Docstring size already present on %%s", doc_cls)
        return False

    def _size(self: DocT) -> int:
        blocks: list[str] = []
        if self.short_description:
            blocks.append(self.short_description)
        if self.long_description:
            blocks.append(self.long_description)
        for entry in self.meta:
            description = entry.description
            if description:
                blocks.append(description)
        return sum(len(block) for block in blocks)

    doc_cls_any = cast(Any, doc_cls)
    doc_cls_any.size = property(_size)
    logger.debug("Registered size property on %%s", doc_cls)
    return True


doc_common: DocstringCommonModuleProto | None = None

try:  # pragma: no cover - best effort compatibility shim
    from docstring_parser import common as _imported_common
except Exception:  # pragma: no cover - the library is optional at runtime
    logger.debug("docstring_parser.common unavailable; skipping compatibility shim")
else:
    doc_common = cast(DocstringCommonModuleProto, _imported_common)

if doc_common is not None and not ENABLE_SITECUSTOMIZE:
    logger.info(
        "Docstring builder shims disabled via KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=%s",
        _FLAG_VALUE or "0",
    )

if doc_common is not None and ENABLE_SITECUSTOMIZE:
    warnings.warn(
        "Docstring builder runtime shims are deprecated and will be removed; "
        "set KGFOUNDRY_DOCSTRINGS_SITECUSTOMIZE=0 to opt out early.",
        DeprecationWarning,
        stacklevel=2,
    )
    doc_common_local = doc_common
    if not hasattr(doc_common_local, "DocstringYields") and hasattr(
        doc_common_local, "DocstringReturns"
    ):
        base_returns: type[DocstringReturnsProto] = doc_common_local.DocstringReturns

        def _init_docstring_yields(
            self: DocstringYieldsProto,
            args: Sequence[str],
            description: str | None,
            type_name: str | None,
            return_name: str | None = None,
        ) -> None:
            base_returns.__init__(
                self,
                list(args),
                description,
                type_name,
                is_generator=True,
                return_name=return_name,
            )
            self.is_generator = True

        shim_doc = (
            "Backward compatible substitute for generator metadata entries.\n\n"
            "``docstring_parser`` introduced ``DocstringYields`` in newer releases.\n"
            "Older versions only expose ``DocstringReturns``.  This dynamically\n"
            "created subclass mimics the modern type so that downstream code can\n"
            "rely on generator-specific helpers regardless of the installed library\n"
            "version."
        )

        _DocstringYieldsShim = cast(
            type[DocstringYieldsProto],
            type(
                "DocstringYields",
                (base_returns,),
                {
                    "__doc__": shim_doc,
                    "__module__": base_returns.__module__,
                    "__init__": _init_docstring_yields,
                },
            ),
        )
        doc_common_local.DocstringYields = _DocstringYieldsShim
        logger.debug("Registered DocstringYields compatibility shim")

    doc_cls: type[DocstringProto] | None = getattr(doc_common_local, "Docstring", None)
    attr_cls_candidate = getattr(doc_common_local, "DocstringAttr", None) or getattr(
        doc_common_local, "DocstringParam", None
    )
    attr_cls = cast(type[DocstringAttrProto] | None, attr_cls_candidate)
    yields_cls: type[DocstringYieldsProto] | None = getattr(
        doc_common_local, "DocstringYields", None
    )

    if doc_cls is not None and attr_cls is not None and ensure_docstring_attrs(doc_cls, attr_cls):
        logger.debug("Docstring.attrs shim installed")

    if doc_cls is not None and yields_cls is not None:
        added_yield, added_many = ensure_docstring_yields(doc_cls, yields_cls)
        if added_yield or added_many:
            logger.debug(
                "Docstring yields shim installed (single=%s many=%s)", added_yield, added_many
            )

    if doc_cls is not None and ensure_docstring_size(doc_cls):
        logger.debug("Docstring size shim installed")

__all__ = [
    "ensure_docstring_attrs",
    "ensure_docstring_size",
    "ensure_docstring_yields",
]
