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
        """Type-checking placeholder for generator metadata entries.

        <!-- auto:docstring-builder v1 -->

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

        ...

else:  # pragma: no cover - imported lazily when type checking is disabled
    Docstring = Any
    DocstringParam = Any
    DocstringReturns = Any
    DocstringYields = Any


class DocstringMetaProto(Protocol):
    """Common surface shared by docstring ``meta`` entries.

    <!-- auto:docstring-builder v1 -->

    ``docstring_parser`` represents return blocks, attributes, and parameter
    annotations as ``meta`` objects.  Each metadata entry exposes a short
    textual description that may be ``None`` when the source docstring omitted
    additional context.  The concrete subclasses supplied by
    ``docstring_parser`` add extra fields, but they all share this description
    surface which is the only part we need for compatibility.

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

    description: str | None


class DocstringAttrProto(DocstringMetaProto, Protocol):
    """Protocol describing attribute metadata entries.

    <!-- auto:docstring-builder v1 -->

    ``docstring_parser`` represents both attributes and parameters using
    :class:`DocstringParam`.  Attribute metadata entries expose their marker and
    attribute name via ``args`` while reusing ``description`` for human readable
    explanations.  We only need the attribute name list to populate the
    ``attrs`` property in the shimmed ``Docstring`` objects.

    Parameters
    ----------
    *args : Sequence[str]
        Describe ``args``.
    **kwargs : inspect._empty
        Describe ``kwargs``.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    args: Sequence[str]


class DocstringReturnsProto(DocstringMetaProto, Protocol):
    """Protocol describing return metadata entries.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    args : Sequence[str]
        Describe ``args``.
    description : str | None
        Describe ``description``.
    type_name : str | None
        Describe ``type_name``.
    is_generator : bool
        Describe ``is_generator``.
    return_name : str | None, optional
        Describe ``return_name``.
        Defaults to ``None``.
    """

    args: Sequence[str]
    type_name: str | None
    is_generator: bool


class DocstringYieldsProto(DocstringReturnsProto, Protocol):
    """Protocol describing generator metadata entries.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    args : Sequence[str]
        Describe ``args``.
    description : str | None
        Describe ``description``.
    type_name : str | None
        Describe ``type_name``.
    is_generator : bool
        Describe ``is_generator``.
    return_name : str | None
        Describe ``return_name``.
    """

    return_name: str | None


class DocstringProto(Protocol):
    """Protocol describing :class:`docstring_parser.common.Docstring`.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    meta : Sequence[DocstringMetaProto]
        Describe ``meta``.
    short_description : str | None
        Describe ``short_description``.
    long_description : str | None
        Describe ``long_description``.
    """

    meta: list[DocstringMetaProto]
    short_description: str | None
    long_description: str | None


class DocstringCommonModuleProto(Protocol):
    """Protocol describing the ``docstring_parser.common`` module surface.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    Docstring : DocstringProto
        Describe ``Docstring``.
    DocstringAttr : DocstringAttrProto
        Describe ``DocstringAttr``.
    DocstringParam : DocstringAttrProto
        Describe ``DocstringParam``.
    DocstringReturns : DocstringReturnsProto
        Describe ``DocstringReturns``.
    """

    Docstring: type[DocstringProto]
    DocstringAttr: type[DocstringAttrProto] | None
    DocstringParam: type[DocstringAttrProto] | None
    DocstringReturns: type[DocstringReturnsProto]
    DocstringYields: type[DocstringYieldsProto] | None


def _get_attr_name(entry: DocstringAttrProto) -> str | None:
    """Return the attribute identifier embedded in a metadata entry."""
    if not entry.args:
        return None
    if len(entry.args) == 1:
        return entry.args[0]
    if entry.args[0] == "attribute":
        return entry.args[1]
    return entry.args[0]


def ensure_docstring_attrs(
    doc_cls: type[DocstringProto], attr_cls: type[DocstringAttrProto]
) -> bool:
    """Install an ``attrs`` property on ``Docstring`` when absent."""
    if hasattr(doc_cls, "attrs"):
        return False

    def _attrs(self: DocstringProto) -> list[DocstringAttrProto]:
        """Return attribute metadata entries collected from Docstring meta."""
        return [
            entry
            for entry in self.meta
            if isinstance(entry, attr_cls) and _get_attr_name(entry) is not None
        ]

    doc_cls_any = cast(Any, doc_cls)
    doc_cls_any.attrs = property(_attrs)
    logger.debug("Registered attrs property on %s", doc_cls)
    return True


def ensure_docstring_yields(
    doc_cls: type[DocstringProto], yields_cls: type[DocstringYieldsProto]
) -> tuple[bool, bool]:
    """Install ``yields`` and ``many_yields`` helpers on ``Docstring``."""
    added_single = False
    added_many = False

    if not hasattr(doc_cls, "yields"):

        def _yield_entry(self: DocstringProto) -> DocstringYieldsProto | None:
            """Return the first generator metadata entry, if any."""
            for entry in self.meta:
                if isinstance(entry, yields_cls):
                    return entry
            return None

        doc_cls_any = cast(Any, doc_cls)
        doc_cls_any.yields = property(_yield_entry)
        added_single = True

    if not hasattr(doc_cls, "many_yields"):

        def _many(self: DocstringProto) -> list[DocstringYieldsProto]:
            """Return all generator metadata entries in the docstring."""
            return [entry for entry in self.meta if isinstance(entry, yields_cls)]

        doc_cls_any = cast(Any, doc_cls)
        doc_cls_any.many_yields = property(_many)
        added_many = True

    if added_single:
        logger.debug("Registered yields property on %s", doc_cls)
    if added_many:
        logger.debug("Registered many_yields property on %s", doc_cls)
    return added_single, added_many


def ensure_docstring_size(doc_cls: type[DocstringProto]) -> bool:
    """Install a ``size`` property mirroring ``Docstring.total_size`` semantics."""
    if hasattr(doc_cls, "size"):
        return False

    def _size(self: DocstringProto) -> int:
        """Return the total length of rendered docstring components."""
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
    logger.debug("Registered size property on %s", doc_cls)
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
            """Describe  init docstring yields.

            <!-- auto:docstring-builder v1 -->

            Special method customising Python's object protocol for this class. Use it to integrate with built-in operators, protocols, or runtime behaviours that expect instances to participate in the language's data model.

            Parameters
            ----------
            args : Sequence
            Describe `args`.
            description : str | None, optional
            Describe `description`.
            type_name : str | None, optional
            Describe `type_name`.
            return_name : str | None, optional, by default None
            Describe `return_name`.
            """
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
