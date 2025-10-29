#!/usr/bin/env python
"""Overview of auto docstrings.

This module bundles auto docstrings logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""

from __future__ import annotations

import argparse
import ast
import inspect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _paragraph(*sentences: str) -> str:
    """Join sentences into a clean paragraph used throughout generated summaries."""
    parts = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
    return " ".join(parts)


def _binary_operator_summary(symbol: str, description: str) -> str:
    """Return an extended summary for binary operator overloads."""
    return _paragraph(
        f"Implement the {description} operator invoked by the `{symbol}` syntax.",
        "Python calls this hook to integrate the instance with arithmetic expressions and expects the result of the operation.",
        "Return ``NotImplemented`` when the other operand cannot participate so Python can fall back to reflected handlers.",
    )


def _reverse_operator_summary(symbol: str, description: str) -> str:
    """Return an extended summary for reflected binary operator overloads."""
    return _paragraph(
        f"Handle reflected {description} when `{symbol}` places this instance on the right-hand side.",
        "Python dispatches here if the left operand cannot finish the operation and needs an alternate implementation.",
        "Return ``NotImplemented`` to allow other types to offer their own reflection strategy when the combination is unsupported.",
    )


def _inplace_operator_summary(symbol: str, description: str) -> str:
    """Return an extended summary for in-place operator overloads."""
    return _paragraph(
        f"Apply in-place {description} using the `{symbol}=` augmented assignment syntax.",
        "Implementations may mutate the instance directly and should return the updated value to mirror Python's expectations.",
        "When in-place mutation is impossible the method should fall back to the normal binary behaviour instead of corrupting state.",
    )


def _unary_operator_summary(symbol: str, description: str) -> str:
    """Return an extended summary for unary operator overloads."""
    return _paragraph(
        f"Implement the unary {description} operation triggered by `{symbol}value` expressions.",
        "Python routes calls from the matching built-in operator to this hook so classes can provide natural mathematical semantics.",
        "Return ``NotImplemented`` or raise ``TypeError`` when the operation is undefined for the instance.",
    )


def _conversion_summary(target: str, usage: str) -> str:
    """Return an extended summary describing a type-conversion special method."""
    return _paragraph(
        f"Provide the instance's {target} representation consumed by {usage}.",
        "Python invokes this hook implicitly when interoperating with built-ins that require the target type.",
        "Raise ``TypeError`` if the value cannot be expressed in that form to signal unsupported conversions clearly.",
    )


def _collection_summary(intent: str, usage: str) -> str:
    """Return an extended summary for collection protocol helpers."""
    return _paragraph(
        intent,
        f"The method lets the instance participate in Python's {usage} protocols without manual iteration helpers.",
        "Implementations should return sensible defaults and respect the expectations of the consuming built-in function.",
    )


def _pydantic_summary(purpose: str, behaviour: str) -> str:
    """Return an extended summary for automatically generated Pydantic helpers."""
    return _paragraph(
        purpose,
        behaviour,
        "Pydantic populates this attribute during model construction, so applications should treat it as read-only metadata.",
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _humanize_identifier(value: str) -> str:
    """Compute humanize identifier.

    Carry out the humanize identifier operation.

    Parameters
    ----------
    value : str
        Description for ``value``.

    Returns
    -------
    str
        Description of return value.
    """
    cleaned = value.replace("_", " ").strip()
    return " ".join(cleaned.split())


def _is_magic(name: str | None) -> bool:
    """Return whether ``name`` is a Python magic method."""
    return bool(name and name.startswith("__") and name.endswith("__"))


def _is_pydantic_model(node: ast.ClassDef) -> bool:
    """Return whether ``node`` inherits from pydantic BaseModel."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


def _is_pydantic_artifact(name: str | None) -> bool:
    """Return ``True`` when ``name`` refers to a Pydantic helper attribute."""
    if not name:
        return False
    return bool(
        name.startswith("model_")
        or name.startswith("__pydantic_")
        or name in PYDANTIC_ARTIFACT_SUMMARIES
        or name
        in {
            "__class_vars__",
            "__private_attributes__",
            "__signature__",
        }
    )


def _is_dataclass_artifact(name: str | None) -> bool:
    """Return ``True`` when ``name`` matches a dataclass support attribute."""
    if not name:
        return False
    return name in {
        "__dataclass_fields__",
        "__dataclass_params__",
        "__match_args__",
        "asdict",
        "astuple",
        "replace",
    }


_CORE_MAGIC_METHOD_SUMMARIES: dict[str, str] = {
    "__repr__": _paragraph(
        "Return an unambiguous string representation primarily used for debugging.",
        "The result should communicate enough state that developers can understand the instance at a glance.",
        "It feeds the ``repr()`` builtin and interactive interpreter output.",
    ),
    "__str__": _paragraph(
        "Provide a readable description of the instance for user-facing output.",
        "This hook powers ``str()`` and print statements, so the message should focus on clarity rather than reproducibility.",
        "Fall back to ``__repr__`` when no friendlier text exists.",
    ),
    "__len__": _paragraph(
        "Report how many items the instance contains.",
        "Containers rely on this count to size allocations and validate expectations.",
        "Return a non-negative integer and raise ``TypeError`` when length is undefined.",
    ),
    "__iter__": _paragraph(
        "Yield each element from the instance in iteration order.",
        "Python calls this hook to support ``for`` loops, comprehensions, and many built-in utilities.",
        "Return an iterator object and raise ``TypeError`` for non-iterable values.",
    ),
    "__aiter__": _paragraph(
        "Produce an asynchronous iterator used by ``async for`` loops.",
        "The hook must return an object implementing ``__anext__`` to cooperate with asynchronous iteration.",
        "Use it to stream data without blocking the event loop.",
    ),
    "__next__": _paragraph(
        "Return the next element from the iterator.",
        "Iteration protocols depend on this hook to drive ``next()`` and ``for`` loops.",
        "Raise ``StopIteration`` to signal exhaustion and avoid silent infinite loops.",
    ),
    "__anext__": _paragraph(
        "Return the next awaitable element from an asynchronous iterator.",
        "``async for`` loops await the result of this hook to progress.",
        "Raise ``StopAsyncIteration`` to end the asynchronous stream cleanly.",
    ),
    "__getitem__": _paragraph(
        "Fetch an element by index or key using subscription syntax.",
        "Implementations translate ``obj[key]`` lookups into domain-specific retrieval logic.",
        "Raise ``KeyError`` or ``IndexError`` when the requested item is absent.",
    ),
    "__setitem__": _paragraph(
        "Store a value associated with a key or index.",
        "This hook supports ``obj[key] = value`` syntax by updating the underlying container.",
        "Validate inputs and raise the appropriate error for unsupported keys.",
    ),
    "__delitem__": _paragraph(
        "Remove the item stored for a specific key or index.",
        "Python calls this hook when executing ``del obj[key]``.",
        "Raise ``KeyError`` or ``IndexError`` if the mapping does not contain the target.",
    ),
    "__contains__": _paragraph(
        "Determine whether a value is present within the instance.",
        "The hook allows ``value in obj`` to execute efficiently without full iteration.",
        "Return a boolean result or fall back to iteration when necessary.",
    ),
    "__bool__": _paragraph(
        "Report whether the instance should evaluate to ``True`` in boolean contexts.",
        "Python defers to this hook for ``if obj`` checks when ``__len__`` is not defined.",
        "Return a strict boolean to avoid surprising truthiness semantics.",
    ),
    "__eq__": _paragraph(
        "Compare the instance for equality with another value.",
        "This hook powers ``==`` comparisons and should consider semantic equivalence rather than identity.",
        "Return ``NotImplemented`` when the other operand is of an incompatible type.",
    ),
    "__ne__": _paragraph(
        "Determine whether two instances differ.",
        "Python falls back to this method when ``__eq__`` does not provide an answer.",
        "Return ``NotImplemented`` for unsupported types so alternate logic can run.",
    ),
    "__lt__": _paragraph(
        "Order the instance relative to another using the less-than comparison.",
        "Sorted containers and algorithms depend on this hook for deterministic ordering.",
        "Return ``NotImplemented`` when comparisons across types make no sense.",
    ),
    "__le__": _paragraph(
        "Implement the less-than-or-equal-to comparison between instances.",
        "It complements ``__lt__`` to support ``<=`` checks in sorting and validation logic.",
        "Return ``NotImplemented`` if the relationship cannot be established for the operands.",
    ),
    "__gt__": _paragraph(
        "Order the instance relative to another using the greater-than operator.",
        "The method mirrors ``__lt__`` but supports ``>`` comparisons.",
        "Return ``NotImplemented`` when types cannot be compared meaningfully.",
    ),
    "__ge__": _paragraph(
        "Implement the greater-than-or-equal-to comparison between values.",
        "It completes the rich comparison set required by ordered containers.",
        "Return ``NotImplemented`` rather than guessing when operands are incompatible.",
    ),
    "__hash__": _paragraph(
        "Produce the hash code associated with the instance.",
        "Hashable objects must provide a stable integer so they can act as dictionary keys or set members.",
        "Ensure the hash is consistent with equality comparisons to avoid lookup bugs.",
    ),
    "__call__": _paragraph(
        "Allow the instance to be invoked like a function.",
        "This hook powers ``obj(...)`` syntax so objects can encapsulate callable behaviour.",
        "Use it to expose the primary operation performed by the instance while preserving internal state.",
    ),
    "__enter__": _paragraph(
        "Prepare resources when entering a context manager.",
        "``with`` statements call this hook before executing the managed block.",
        "Return the object or resource handed to the block's ``as`` target.",
    ),
    "__exit__": _paragraph(
        "Release resources when leaving a context manager.",
        "Python passes exception details here so implementations can decide whether to swallow failures.",
        "Return ``True`` to suppress the exception or ``False`` to propagate it.",
    ),
    "__aenter__": _paragraph(
        "Establish asynchronous resources when entering an async context manager.",
        "``async with`` statements await this hook to prepare the managed value.",
        "Return the object yielded to the context block once setup is complete.",
    ),
    "__aexit__": _paragraph(
        "Clean up asynchronous resources when leaving an async context manager.",
        "The hook mirrors ``__exit__`` but executes within the event loop and may await cleanup operations.",
        "Return ``True`` to silence exceptions or ``False`` to re-raise them.",
    ),
    "__await__": _paragraph(
        "Provide an awaitable interface for the instance.",
        "``await obj`` delegates to this hook, allowing custom objects to model asynchronous computations.",
        "Return an iterator yielding awaitable steps so the event loop can resume the coroutine.",
    ),
    "__copy__": _paragraph(
        "Produce a shallow copy of the instance.",
        "The ``copy`` module calls this hook to duplicate container structures without deep recursion.",
        "Ensure the new object references shared members appropriately for performance.",
    ),
    "__deepcopy__": _paragraph(
        "Create a deep copy of the instance and its nested members.",
        "Used by ``copy.deepcopy`` to recursively duplicate complex graphs while preserving referential integrity.",
        "Respect the ``memo`` dictionary to avoid infinite loops with cyclic references.",
    ),
}


_OBJECT_LIFECYCLE_MAGIC_METHODS: dict[str, str] = {
    "__new__": _paragraph(
        "Allocate and initialise a new instance before ``__init__`` executes.",
        "Override this method when custom memory layout or immutable construction steps are required.",
        "Return the freshly created object, usually by delegating to ``super().__new__``.",
    ),
    "__del__": _paragraph(
        "Run finalisation logic when the garbage collector is about to destroy the instance.",
        "Use this hook for best-effort cleanup of external resources that are not handled by context managers.",
        "It may never execute, so critical teardown should live elsewhere.",
    ),
    "__init_subclass__": _paragraph(
        "Customise subclass creation when derived types are defined.",
        "Python calls this hook on the parent class, enabling registration or attribute validation for new subclasses.",
        "Override it to enforce interfaces or auto-configure mixins during inheritance.",
    ),
}


_ATTRIBUTE_ACCESS_MAGIC_METHODS: dict[str, str] = {
    "__getattr__": _paragraph(
        "Provide a fallback attribute lookup when normal resolution fails.",
        "The hook enables dynamic attributes and lazy loading for missing names.",
        "Raise ``AttributeError`` to preserve Python's attribute access semantics.",
    ),
    "__getattribute__": _paragraph(
        "Intercept every attribute access on the instance.",
        "Override this hook to implement global logging, proxies, or computed attributes.",
        "Call ``super().__getattribute__`` for standard behaviour to avoid infinite recursion.",
    ),
    "__setattr__": _paragraph(
        "Control how attributes are assigned on the instance.",
        "It underpins ``obj.name = value`` syntax and enables validation or transformation before storage.",
        "Delegate to ``super().__setattr__`` when default assignment is sufficient.",
    ),
    "__delattr__": _paragraph(
        "Handle attribute deletion requests.",
        "Python invokes this method for ``del obj.name`` so classes can manage cleanup or forbid removal.",
        "Raise ``AttributeError`` to signal that the attribute cannot be deleted.",
    ),
    "__dir__": _paragraph(
        "Return the ordered collection of attribute names visible on the instance.",
        "Interactive tools and auto-completion engines call this hook when exploring objects.",
        "Include dynamic attributes so users discover extension points easily.",
    ),
}


_DESCRIPTOR_MAGIC_METHODS: dict[str, str] = {
    "__get__": _paragraph(
        "Implement the descriptor protocol for attribute retrieval.",
        "Python routes ``instance.attr`` access through this hook when the class defines a descriptor.",
        "Return the computed value or the descriptor itself when accessed on the class.",
    ),
    "__set__": _paragraph(
        "Handle assignments to attributes managed by a descriptor.",
        "This hook enables validation and custom storage when ``instance.attr = value`` executes.",
        "Raise ``AttributeError`` to forbid updates on read-only descriptors.",
    ),
    "__delete__": _paragraph(
        "Process deletion of attributes bound to a descriptor.",
        "Descriptors can use this hook to purge cached state or raise errors for immutable properties.",
        "Implement it alongside ``__set__`` to complete the management contract.",
    ),
    "__set_name__": _paragraph(
        "Receive the attribute name when a descriptor is assigned on a class definition.",
        "Python calls this hook once during class creation to provide context about where the descriptor lives.",
        "Use the information to register the descriptor or compute derived metadata.",
    ),
}


_PICKLING_MAGIC_METHODS: dict[str, str] = {
    "__getstate__": _paragraph(
        "Prepare the serialisable state for pickling.",
        "This hook allows classes to prune or transform attributes before the pickle module persists them.",
        "Return an object understood by ``__setstate__`` during restoration.",
    ),
    "__setstate__": _paragraph(
        "Restore the instance from the state produced by ``__getstate__`` or the pickler.",
        "The method receives the persisted data and should reinitialise transient resources.",
        "Use it to maintain backward compatibility when schemas evolve.",
    ),
    "__reduce__": _paragraph(
        "Describe how to reconstruct the instance for older pickle protocols.",
        "Return a tuple detailing the callable and arguments needed to recreate the object.",
        "Fallback to ``__reduce_ex__`` for protocol-specific behaviour when required.",
    ),
    "__reduce_ex__": _paragraph(
        "Provide protocol-aware pickling instructions.",
        "The pickle module passes the protocol version so classes can adjust their reconstruction strategy.",
        "Delegate to ``__reduce__`` when custom handling is unnecessary.",
    ),
    "__getnewargs__": _paragraph(
        "Return positional arguments passed to ``__new__`` during unpickling.",
        "Immutable types use this hook to ensure construction receives the proper initial values.",
        "Pair it with ``__getstate__`` to restore complete object state.",
    ),
    "__getnewargs_ex__": _paragraph(
        "Return positional and keyword arguments required by ``__new__`` during unpickling.",
        "This extended form supports pickle protocols that separate args and kwargs for complex constructors.",
        "Use it when simple positional arguments cannot rebuild the instance.",
    ),
}


_BINARY_OPERATOR_MAGIC_METHODS: dict[str, str] = {
    "__add__": _binary_operator_summary("+", "addition"),
    "__sub__": _binary_operator_summary("-", "subtraction"),
    "__mul__": _binary_operator_summary("*", "multiplication"),
    "__matmul__": _binary_operator_summary("@", "matrix multiplication"),
    "__truediv__": _binary_operator_summary("/", "true division"),
    "__floordiv__": _binary_operator_summary("//", "floor division"),
    "__mod__": _binary_operator_summary("%", "modulo"),
    "__divmod__": _binary_operator_summary("divmod", "combined division and modulo"),
    "__pow__": _binary_operator_summary("**", "exponentiation"),
    "__lshift__": _binary_operator_summary("<<", "left bitwise shift"),
    "__rshift__": _binary_operator_summary(">>", "right bitwise shift"),
    "__and__": _binary_operator_summary("&", "bitwise AND"),
    "__xor__": _binary_operator_summary("^", "bitwise XOR"),
    "__or__": _binary_operator_summary("|", "bitwise OR"),
}


_REVERSE_OPERATOR_MAGIC_METHODS: dict[str, str] = {
    "__radd__": _reverse_operator_summary("+", "addition"),
    "__rsub__": _reverse_operator_summary("-", "subtraction"),
    "__rmul__": _reverse_operator_summary("*", "multiplication"),
    "__rmatmul__": _reverse_operator_summary("@", "matrix multiplication"),
    "__rtruediv__": _reverse_operator_summary("/", "true division"),
    "__rfloordiv__": _reverse_operator_summary("//", "floor division"),
    "__rmod__": _reverse_operator_summary("%", "modulo"),
    "__rdivmod__": _reverse_operator_summary("divmod", "combined division and modulo"),
    "__rpow__": _reverse_operator_summary("**", "exponentiation"),
    "__rlshift__": _reverse_operator_summary("<<", "left bitwise shift"),
    "__rrshift__": _reverse_operator_summary(">>", "right bitwise shift"),
    "__rand__": _reverse_operator_summary("&", "bitwise AND"),
    "__rxor__": _reverse_operator_summary("^", "bitwise XOR"),
    "__ror__": _reverse_operator_summary("|", "bitwise OR"),
}


_INPLACE_OPERATOR_MAGIC_METHODS: dict[str, str] = {
    "__iadd__": _inplace_operator_summary("+", "addition"),
    "__isub__": _inplace_operator_summary("-", "subtraction"),
    "__imul__": _inplace_operator_summary("*", "multiplication"),
    "__imatmul__": _inplace_operator_summary("@", "matrix multiplication"),
    "__itruediv__": _inplace_operator_summary("/", "true division"),
    "__ifloordiv__": _inplace_operator_summary("//", "floor division"),
    "__imod__": _inplace_operator_summary("%", "modulo"),
    "__ipow__": _inplace_operator_summary("**", "exponentiation"),
    "__ilshift__": _inplace_operator_summary("<<", "left bitwise shift"),
    "__irshift__": _inplace_operator_summary(">>", "right bitwise shift"),
    "__iand__": _inplace_operator_summary("&", "bitwise AND"),
    "__ixor__": _inplace_operator_summary("^", "bitwise XOR"),
    "__ior__": _inplace_operator_summary("|", "bitwise OR"),
}


_UNARY_OPERATOR_MAGIC_METHODS: dict[str, str] = {
    "__neg__": _unary_operator_summary("-", "negation"),
    "__pos__": _unary_operator_summary("+", "unary plus"),
    "__abs__": _paragraph(
        "Compute the absolute value of the instance.",
        "This hook powers ``abs(value)`` so numeric types can describe magnitude.",
        "Return a non-negative representation or raise ``TypeError`` when undefined.",
    ),
    "__invert__": _unary_operator_summary("~", "bitwise inversion"),
}


_TYPE_CONVERSION_MAGIC_METHODS: dict[str, str] = {
    "__int__": _conversion_summary("integer", "integer arithmetic and built-ins like ``int()``"),
    "__float__": _conversion_summary("floating point", "``float()`` and numeric coercion"),
    "__complex__": _conversion_summary("complex", "the ``complex()`` constructor"),
    "__index__": _paragraph(
        "Provide an exact integer representation suitable for slicing and bit operations.",
        "Python relies on this hook for ``hex()``, ``bin()``, and built-ins that require an index value.",
        "Return a non-negative integer to remain compatible with array protocols.",
    ),
    "__round__": _paragraph(
        "Implement rounding behaviour for the instance.",
        "Built-ins call this hook to compute ``round(value, ndigits)`` with domain-specific precision rules.",
        "Respect the optional ``ndigits`` argument when provided to mirror numeric types.",
    ),
    "__trunc__": _paragraph(
        "Return the truncated integer value of the instance.",
        "The ``math.trunc`` function delegates to this hook for custom numeric types.",
        "Prefer returning an ``int`` so downstream consumers can rely on standard behaviour.",
    ),
    "__floor__": _paragraph(
        "Return the greatest integer less than or equal to the value.",
        "``math.floor`` invokes this hook when available to avoid redundant conversions.",
        "Ensure the result honours mathematical floor semantics for fractional values.",
    ),
    "__ceil__": _paragraph(
        "Return the smallest integer greater than or equal to the value.",
        "``math.ceil`` dispatches here to respect object-specific rounding rules.",
        "Return an ``int`` or compatible numeric type to preserve expectations.",
    ),
}


_COLLECTION_MAGIC_METHODS: dict[str, str] = {
    "__reversed__": _collection_summary(
        "Produce an iterator that yields elements in reverse order.",
        "reverse iteration via ``reversed()``",
    ),
    "__length_hint__": _collection_summary(
        "Provide an expected number of items that iteration will produce.",
        "iterator preallocation and optimisation",
    ),
    "__missing__": _collection_summary(
        "Handle missing dictionary keys when subclasses of ``dict`` cannot find an entry.",
        "mapping lookup error handling",
    ),
}


_TYPE_SYSTEM_MAGIC_METHODS: dict[str, str] = {
    "__instancecheck__": _paragraph(
        "Decide whether an object should be considered an instance of the class for ``isinstance``.",
        "Metaclasses override this hook to implement virtual subclassing or structural typing.",
        "Return ``True`` or ``False`` to guide runtime type checks.",
    ),
    "__subclasscheck__": _paragraph(
        "Determine whether another class should be treated as a subclass for ``issubclass``.",
        "Use this hook to support plug-in architectures or ABC-style registration.",
        "Return booleans and avoid expensive operations to keep type checks fast.",
    ),
    "__class_getitem__": _paragraph(
        "Enable subscription on the class object itself, commonly for generics.",
        "Python passes the bracketed arguments here so classes can manufacture specialised variants.",
        "Return a new class or metadata describing the parametrised type.",
    ),
}


_MISCELLANEOUS_MAGIC_METHODS: dict[str, str] = {
    "__bytes__": _paragraph(
        "Return a ``bytes`` representation of the instance.",
        "The ``bytes`` constructor delegates here to serialise custom data structures.",
        "Provide a stable encoding or raise ``TypeError`` when conversion is not meaningful.",
    ),
    "__format__": _paragraph(
        "Customise how the instance renders within formatted strings.",
        "Python calls this method from ``format`` and f-strings to honour format specifications.",
        "Use it to expose alignment, precision, or style controls relevant to the object.",
    ),
    "__sizeof__": _paragraph(
        "Report the memory footprint of the instance in bytes.",
        "The ``sys.getsizeof`` helper calls this method to gather sizing information.",
        "Include the size of referenced buffers when possible to keep reports accurate.",
    ),
    "__fspath__": _paragraph(
        "Return the filesystem path representation of the object.",
        "``os.fspath`` uses this hook so path-like objects can integrate with file APIs.",
        "Return a ``str`` or ``bytes`` path to remain compatible with the filesystem layer.",
    ),
    "__buffer__": _paragraph(
        "Expose a writable buffer interface to the instance's underlying memory.",
        "Python's buffer protocol queries this hook when consumers request direct byte access.",
        "Return a memoryview-compatible object or raise ``TypeError`` if the buffer cannot be shared.",
    ),
    "__release_buffer__": _paragraph(
        "Release resources acquired for a previous ``__buffer__`` request.",
        "The buffer protocol calls this hook when a consumer finishes accessing exported memory.",
        "Clean up state associated with the buffer view to prevent leaks.",
    ),
}


_STANDARD_METHOD_EXTENDED_SUMMARIES: dict[str, str] = {
    "clear": _paragraph(
        "Remove every entry from the mapping so callers can start from a clean slate.",
        "The helper mirrors ``dict.clear`` so code interacting with mapping-like models behaves consistently.",
        "Return ``None`` to match the built-in contract and avoid implying a meaningful value.",
    ),
    "copy": _paragraph(
        "Return a shallow copy of the mapping suitable for defensive mutation.",
        "Consumers can tweak the duplicate without affecting the original instance or violating validation rules.",
        "Prefer this helper over direct constructors to preserve model-specific behaviours.",
    ),
    "get": _paragraph(
        "Retrieve a value for ``key`` while falling back to a default when absent.",
        "The convenience wrapper mirrors ``dict.get`` so configuration objects remain ergonomic.",
        "Use it to express optional access without raising ``KeyError``.",
    ),
    "items": _collection_summary(
        "Return a dynamic view exposing each ``(key, value)`` pair.",
        "mapping iteration and dictionary-style inspection",
    ),
    "keys": _collection_summary(
        "Expose a set-like view of the mapping's keys.",
        "mapping introspection and membership checks",
    ),
    "values": _collection_summary(
        "Provide a view of the mapping's stored values.",
        "mapping iteration utilities",
    ),
}


MAGIC_METHOD_EXTENDED_SUMMARIES: dict[str, str] = {
    **_CORE_MAGIC_METHOD_SUMMARIES,
    **_OBJECT_LIFECYCLE_MAGIC_METHODS,
    **_ATTRIBUTE_ACCESS_MAGIC_METHODS,
    **_DESCRIPTOR_MAGIC_METHODS,
    **_PICKLING_MAGIC_METHODS,
    **_BINARY_OPERATOR_MAGIC_METHODS,
    **_REVERSE_OPERATOR_MAGIC_METHODS,
    **_INPLACE_OPERATOR_MAGIC_METHODS,
    **_UNARY_OPERATOR_MAGIC_METHODS,
    **_TYPE_CONVERSION_MAGIC_METHODS,
    **_COLLECTION_MAGIC_METHODS,
    **_TYPE_SYSTEM_MAGIC_METHODS,
    **_MISCELLANEOUS_MAGIC_METHODS,
}


DEFAULT_MAGIC_METHOD_FALLBACK = _paragraph(
    "Special method customising Python's object protocol for this class.",
    "It allows instances to plug into built-in syntax or behaviours even when a dedicated handler is not defined.",
    "Return ``NotImplemented`` when the protocol is unsupported so Python can search for alternative implementations.",
)


DEFAULT_PYDANTIC_ARTIFACT_SUMMARY = _paragraph(
    "Internal helper generated by Pydantic for model configuration or validation.",
    "Attributes with this naming pattern coordinate validators, serializers, and schema metadata behind the scenes.",
    "Treat them as implementation details rather than public extension points.",
)


PYDANTIC_ARTIFACT_SUMMARIES: dict[str, str] = {
    "model_config": _pydantic_summary(
        "Expose configuration controlling validation, serialisation, and runtime behaviour.",
        "Pydantic merges class-level declarations into this mapping before constructing the model class.",
    ),
    "model_fields": _pydantic_summary(
        "Describe each field defined on the model, including metadata and validators.",
        "The dictionary maps attribute names to ``FieldInfo`` structures generated during class creation.",
    ),
    "model_computed_fields": _pydantic_summary(
        "Track computed field descriptors declared with ``@computed_field``.",
        "Entries provide lazy evaluation hooks used when serialising or exporting the model.",
    ),
    "model_fields_set": _pydantic_summary(
        "Record which fields were explicitly provided when the instance was initialised.",
        "Validation populates this set so business logic can distinguish defaults from user input.",
    ),
    "model_extra": _pydantic_summary(
        "Store attributes captured by ``extra='allow'`` or similar configuration.",
        "Pydantic keeps these values separate from standard fields to prevent accidental schema drift.",
    ),
    "__class_vars__": _pydantic_summary(
        "Expose class variable annotations preserved on the model for introspection.",
        "Entries reflect ``ClassVar`` declarations so tooling can distinguish runtime fields from static metadata.",
    ),
    "model_post_init": _pydantic_summary(
        "Provide a hook executed after ``__init__`` completes validation.",
        "The callable receives the model instance and allows custom post-processing before it is returned.",
    ),
    "model_rebuild": _pydantic_summary(
        "Trigger recompilation of the model's internal schema cache.",
        "Call this helper when forward references or configuration changes require refreshing the generated schema.",
    ),
    "model_parametrized_name": _pydantic_summary(
        "Return the dynamically generated display name for parametrised models.",
        "Generics rely on this helper to produce stable identifiers for documentation and serialisation.",
    ),
    "model_dump": _pydantic_summary(
        "Serialise the model instance into a plain Python mapping.",
        "Options allow callers to control inclusion, exclusion, and alias handling during export.",
    ),
    "model_dump_json": _pydantic_summary(
        "Render the model as a JSON string using the configured encoders.",
        "It wraps ``model_dump`` to apply JSON-specific serialisation rules and ensure UTF-8 output.",
    ),
    "model_validate": _pydantic_summary(
        "Construct a model instance from raw input data.",
        "This classmethod applies parsing, validation, and conversion before returning the populated object.",
    ),
    "model_validate_json": _pydantic_summary(
        "Parse JSON input and validate it against the model schema.",
        "Internally it deserialises the string then delegates to ``model_validate`` for field coercion.",
    ),
    "model_copy": _pydantic_summary(
        "Create a shallow or deep copy of the instance.",
        "Callers can adjust include or exclude options to copy only selected fields.",
    ),
    "model_construct": _pydantic_summary(
        "Instantiate the model without running validation.",
        "This low-level helper is useful when data is already trusted and performance is critical.",
    ),
    "model_serializer": _pydantic_summary(
        "Return the serializer callable registered for the model.",
        "Custom serializers can modify outbound representations during export workflows.",
    ),
    "model_json_schema": _pydantic_summary(
        "Generate a JSON Schema document describing the model structure.",
        "The method compiles core metadata into a dictionary compatible with schema tooling.",
    ),
    "schema": _pydantic_summary(
        "Produce a legacy JSON-compatible schema for the model.",
        "It exists for backwards compatibility and proxies to ``model_json_schema`` in modern versions.",
    ),
    "schema_json": _pydantic_summary(
        "Return the generated schema as a JSON string.",
        "Callers receive a ready-to-serialise payload for documentation or storage.",
    ),
    "dict": _pydantic_summary(
        "Convert the model into a ``dict`` of field values.",
        "This instance method mirrors ``model_dump`` but defaults to returning Python primitives for nested models.",
    ),
    "json": _pydantic_summary(
        "Serialise the instance to JSON text.",
        "It respects include and exclude options so APIs can shape their payloads precisely.",
    ),
    "copy": _pydantic_summary(
        "Duplicate the model instance.",
        "Behaviour matches ``model_copy`` while preserving backward compatibility with Pydantic 1 style code.",
    ),
    "__pydantic_core_schema__": _pydantic_summary(
        "Store the low-level schema object produced by Pydantic's core runtime.",
        "Advanced extensions introspect this structure to customise validation.",
    ),
    "__pydantic_core_config__": _pydantic_summary(
        "Persist the core configuration driving validation decisions.",
        "It reflects merged settings after Pydantic resolves inheritance and model_config overrides.",
    ),
    "__pydantic_decorators__": _pydantic_summary(
        "Registry tracking validators, root validators, and serializers declared on the model.",
        "The mapping ensures decorators are applied in the correct order during validation and serialisation.",
    ),
    "__pydantic_extra__": _pydantic_summary(
        "Hold arbitrary attributes permitted by the model configuration.",
        "Values appear here when ``extra`` behaviour allows storing keys beyond the declared schema.",
    ),
    "__pydantic_complete__": _pydantic_summary(
        "Flag whether Pydantic finished constructing the model class and its helpers.",
        "Consumers can guard against premature access to partially initialised models using this boolean.",
    ),
    "__pydantic_computed_fields__": _pydantic_summary(
        "Track computed field definitions attached to the model.",
        "The mapping stores decorator metadata so serialisation honours lazily evaluated properties.",
    ),
    "__pydantic_fields_set__": _pydantic_summary(
        "Track which fields were provided to the constructor.",
        "This mirrors ``model_fields_set`` but lives on the instance for quick access.",
    ),
    "__pydantic_parent_namespace__": _pydantic_summary(
        "Store the namespace used to resolve forward references.",
        "It gives validators access to surrounding modules when type hints reference future definitions.",
    ),
    "__pydantic_generic_metadata__": _pydantic_summary(
        "Record metadata about how a generic model was specialised.",
        "The structure captures parameter substitutions so schema generation remains accurate.",
    ),
    "__pydantic_model_complete__": _pydantic_summary(
        "Flag whether Pydantic finished configuring the model class.",
        "The boolean guards against premature access before validators and schema helpers are ready.",
    ),
    "__pydantic_serializer__": _pydantic_summary(
        "Reference the compiled serializer callable for the model.",
        "It is used when exporting data through ``model_dump`` or ``model_dump_json``.",
    ),
    "__pydantic_validator__": _pydantic_summary(
        "Reference the compiled validator callable for the model.",
        "Pydantic executes this entry to transform raw data into model instances efficiently.",
    ),
    "__pydantic_custom_init__": _pydantic_summary(
        "Indicate that the model defines a custom ``__init__`` implementation.",
        "The flag informs internals to respect user-defined constructors during validation.",
    ),
    "__pydantic_private__": _pydantic_summary(
        "Store private attributes declared with ``PrivateAttr``.",
        "Private values live outside the public schema yet remain accessible on the instance.",
    ),
    "__pydantic_root_model__": _pydantic_summary(
        "Describe configuration applied when the model uses the root-model pattern.",
        "It captures type information so validation and schema generation remain aligned with the wrapped value.",
    ),
    "__pydantic_setattr_handlers__": _pydantic_summary(
        "Maintain the compiled attribute-assignment hooks for the model.",
        "Pydantic uses these handlers to enforce validators and field protections during runtime mutation.",
    ),
    "__pydantic_init_subclass__": _pydantic_summary(
        "Provide Pydantic's subclass initialisation helper.",
        "The hook wraps ``__init_subclass__`` to ensure generated models maintain validation metadata.",
    ),
    "__pydantic_post_init__": _pydantic_summary(
        "Point to the function executed immediately after ``__init__`` finishes.",
        "It mirrors ``model_post_init`` but is stored under a Pydantic-reserved name for internal scheduling.",
    ),
    "__private_attributes__": _pydantic_summary(
        "Expose private attribute descriptors declared on the model.",
        "The mapping mirrors ``__pydantic_private__`` but tracks definitions at the class level for introspection.",
    ),
    "__get_pydantic_core_schema__": _pydantic_summary(
        "Compute the core schema for custom types or dataclasses.",
        "Custom model components implement this method so Pydantic can integrate them into validation.",
    ),
    "__get_pydantic_json_schema__": _pydantic_summary(
        "Customise JSON Schema generation for user-defined types.",
        "Implementations adapt the default schema based on context provided by Pydantic.",
    ),
    "__signature__": _pydantic_summary(
        "Expose the generated call signature for the model's constructor.",
        "Introspection tools rely on the ``inspect``-style signature to surface parameters and defaults accurately.",
    ),
}


QUALIFIED_NAME_OVERRIDES: dict[str, str] = {
    # === Project-specific vector store aliases ===
    "FloatArray": "src.vectorstore_faiss.gpu.FloatArray",
    "IntArray": "src.vectorstore_faiss.gpu.IntArray",
    "StrArray": "src.vectorstore_faiss.gpu.StrArray",
    "VecArray": "src.search_api.faiss_adapter.VecArray",
    # === Project-specific client helpers ===
    "_SupportsHttp": "src.search_client.client._SupportsHttp",
    "_SupportsResponse": "src.search_client.client._SupportsResponse",
    # === Project-specific models ===
    "Chunk": "src.kgfoundry_common.models.Chunk",
    "Concept": "src.ontology.catalog.Concept",
    "ConceptMeta": "src.ontology.catalog.ConceptMeta",
    "DatasetVersion": "src.kgfoundry_common.parquet_io.DatasetVersion",
    "Doc": "src.kgfoundry_common.models.Doc",
    "DoctagsAsset": "src.kgfoundry_common.models.DoctagsAsset",
    "FixtureDoc": "src.search_api.fixture_index.FixtureDoc",
    "Id": "src.kgfoundry_common.models.Id",
    "kgfoundry.kgfoundry_common.models.Chunk": "src.kgfoundry_common.models.Chunk",
    "kgfoundry.kgfoundry_common.models.Concept": "src.ontology.catalog.Concept",
    "kgfoundry.kgfoundry_common.models.Doc": "src.kgfoundry_common.models.Doc",
    "kgfoundry.kgfoundry_common.models.DoctagsAsset": "src.kgfoundry_common.models.DoctagsAsset",
    "kgfoundry.kgfoundry_common.models.LinkAssertion": "src.kgfoundry_common.models.LinkAssertion",
    "LinkAssertion": "src.kgfoundry_common.models.LinkAssertion",
    "NavMap": "src.kgfoundry_common.navmap_types.NavMap",
    "Neo4jStore": "src.kg_builder.neo4j_store.Neo4jStore",
    "ParsedSchema": "src.kgfoundry_common.parquet_io.ParsedSchema",
    "SearchRequest": "src.search_api.schemas.SearchRequest",
    "SearchResult": "src.search_api.schemas.SearchResult",
    "SpladeDoc": "src.search_api.splade_index.SpladeDoc",
    # === Project-specific errors ===
    "DownloadError": "src.kgfoundry_common.errors.DownloadError",
    "UnsupportedMIMEError": "src.kgfoundry_common.errors.UnsupportedMIMEError",
    # === Project-specific embedding abstractions ===
    "DenseEmbeddingModel": "src.embeddings_dense.base.DenseEmbeddingModel",
    "SparseEncoder": "src.embeddings_sparse.base.SparseEncoder",
    "SparseIndex": "src.embeddings_sparse.base.SparseIndex",
    # === Pydantic base models ===
    "BaseModel": "pydantic.BaseModel",
    "pydantic.BaseModel": "pydantic.BaseModel",
    # === NumPy core ndarray and typing utilities ===
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.typing.NDArray",
    "numpy.dtype": "numpy.dtype",
    "numpy.ndarray": "numpy.ndarray",
    "numpy.random.Generator": "numpy.random.Generator",
    "numpy.typing.ArrayLike": "numpy.typing.ArrayLike",
    "numpy.typing.NDArray": "numpy.typing.NDArray",
    "np.dtype": "numpy.dtype",
    "np.ndarray": "numpy.ndarray",
    "np.typing.NDArray": "numpy.typing.NDArray",
    # === NumPy scalar types (fully qualified) ===
    "numpy.complex128": "numpy.complex128",
    "numpy.complex64": "numpy.complex64",
    "numpy.float16": "numpy.float16",
    "numpy.float32": "numpy.float32",
    "numpy.float64": "numpy.float64",
    "numpy.int16": "numpy.int16",
    "numpy.int32": "numpy.int32",
    "numpy.int64": "numpy.int64",
    "numpy.int8": "numpy.int8",
    "numpy.uint16": "numpy.uint16",
    "numpy.uint32": "numpy.uint32",
    "numpy.uint64": "numpy.uint64",
    "numpy.uint8": "numpy.uint8",
    # === NumPy scalar types (short aliases) ===
    "np.complex128": "numpy.complex128",
    "np.complex64": "numpy.complex64",
    "np.float16": "numpy.float16",
    "np.float32": "numpy.float32",
    "np.float64": "numpy.float64",
    "np.int16": "numpy.int16",
    "np.int32": "numpy.int32",
    "np.int64": "numpy.int64",
    "np.int8": "numpy.int8",
    "np.uint16": "numpy.uint16",
    "np.uint32": "numpy.uint32",
    "np.uint64": "numpy.uint64",
    "np.uint8": "numpy.uint8",
    # === PyArrow core types ===
    "pyarrow.Array": "pyarrow.Array",
    "pyarrow.DataType": "pyarrow.DataType",
    "pyarrow.Field": "pyarrow.Field",
    "pyarrow.Int64Type": "pyarrow.Int64Type",
    "pyarrow.RecordBatch": "pyarrow.RecordBatch",
    "pyarrow.Schema": "pyarrow.Schema",
    "pyarrow.StringType": "pyarrow.StringType",
    "pyarrow.Table": "pyarrow.Table",
    "pyarrow.TimestampType": "pyarrow.TimestampType",
    "pyarrow.schema": "pyarrow.schema",
    # === Pydantic helpers and validators ===
    "pydantic.AliasChoices": "pydantic.AliasChoices",
    "pydantic.ConfigDict": "pydantic.ConfigDict",
    "pydantic.Field": "pydantic.Field",
    "pydantic.TypeAdapter": "pydantic.TypeAdapter",
    "pydantic.ValidationError": "pydantic.ValidationError",
    "pydantic.field_validator": "pydantic.field_validator",
    "pydantic.fields.Field": "pydantic.fields.Field",
    "pydantic.model_validator": "pydantic.model_validator",
    # === typing and typing_extensions utilities ===
    "Annotated": "typing.Annotated",
    "Any": "typing.Any",
    "Callable": "collections.abc.Callable",
    "Final": "typing.Final",
    "Iterable": "collections.abc.Iterable",
    "Iterator": "collections.abc.Iterator",
    "Literal": "typing.Literal",
    "Mapping": "collections.abc.Mapping",
    "MutableMapping": "collections.abc.MutableMapping",
    "MutableSequence": "collections.abc.MutableSequence",
    "Optional": "typing.Optional",
    "Sequence": "collections.abc.Sequence",
    "Set": "collections.abc.Set",
    "Type": "typing.Type",
    "typing_extensions.Annotated": "typing_extensions.Annotated",
    "typing_extensions.NotRequired": "typing_extensions.NotRequired",
    "typing_extensions.Self": "typing_extensions.Self",
    "typing_extensions.TypeAlias": "typing_extensions.TypeAlias",
    "typing_extensions.TypedDict": "typing_extensions.TypedDict",
    # === Standard library collections and utilities ===
    "collections.Counter": "collections.Counter",
    "collections.defaultdict": "collections.defaultdict",
    "collections.deque": "collections.deque",
    "collections.OrderedDict": "collections.OrderedDict",
    "datetime.date": "datetime.date",
    "datetime.datetime": "datetime.datetime",
    "datetime.timedelta": "datetime.timedelta",
    "pathlib.Path": "pathlib.Path",
    "pathlib.PurePath": "pathlib.PurePath",
    "uuid.UUID": "uuid.UUID",
    # === External service integrations ===
    "duckdb.DuckDBPyConnection": "duckdb.DuckDBPyConnection",
    "Exit": "typer.Exit",
    "fastapi.HTTPException": "fastapi.HTTPException",
    "HTTPException": "fastapi.HTTPException",
    "typer.Exit": "typer.Exit",
}


def _split_generic_arguments(text: str) -> list[str]:
    """Return a list of comma-separated generic arguments respecting nesting."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        if char == "[":
            depth += 1
        elif char == "]" and depth:
            depth -= 1
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_qualified_name(text: str) -> str:
    """Map an annotation string to a fully qualified name when possible."""
    cleaned = text.strip()
    override = QUALIFIED_NAME_OVERRIDES.get(cleaned)
    if override:
        return override
    if "[" in cleaned and cleaned.endswith("]"):
        base, remainder = cleaned.split("[", 1)
        base = base.strip()
        normalized_base = QUALIFIED_NAME_OVERRIDES.get(base, base)
        inner = remainder[:-1]
        arguments = _split_generic_arguments(inner)
        normalized_args = [_normalize_qualified_name(argument) for argument in arguments]
        if QUALIFIED_NAME_OVERRIDES.get(base):
            # Sphinx cannot resolve cross-references that include generic
            # parameters, so we fall back to the canonical base name.
            return normalized_base
        joined = ", ".join(normalized_args) if normalized_args else ""
        if joined:
            return f"{normalized_base}[{joined}]"
        return normalized_base
    return cleaned


def _format_annotation_string(value: str) -> str:
    """Normalise an annotation string for consistent documentation output."""
    text = value.strip().replace("typing.", "")
    replacements = {"list": "List", "dict": "Mapping[str, Any]", "tuple": "Tuple", "set": "Set"}
    if text in replacements:
        text = replacements[text]
    text = text.replace("list[", "List[").replace("tuple[", "Tuple[").replace("set[", "Set[")
    if text.startswith("dict["):
        text = text.replace("dict[", "Mapping[")
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        inner_text = _normalize_qualified_name(_format_annotation_string(inner))
        return f"Optional[{inner_text}]"
    text = _normalize_qualified_name(text)
    if text.startswith("Optional[") and text.endswith("]"):
        inner = text[len("Optional[") : -1]
        inner_text = _normalize_qualified_name(_format_annotation_string(inner))
        return f"Optional[{inner_text}]"
    return text


def _annotation_accepts_none(text: str) -> bool:
    """Return ``True`` when ``text`` indicates that ``None`` is an allowed value."""
    cleaned = text.replace(" ", "")
    if not cleaned:
        return False
    if cleaned == "None":
        return True
    if cleaned.startswith("Optional[") and cleaned.endswith("]"):
        return True
    return "|None" in cleaned or "None|" in cleaned


def _unparse_or_none(node: ast.AST | None) -> str | None:
    """Safely call :func:`ast.unparse` and return ``None`` if it fails."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return None


_DEFAULT_SENTINEL = object()


@dataclass
class ParameterDetails:
    """Describe the metadata captured for a function parameter."""

    name: str
    annotation: str
    original_annotation: str | None = None
    has_default: bool = False
    default_value: Any = _DEFAULT_SENTINEL
    default_text: str | None = None
    accepts_none: bool = False


@dataclass
class DocstringChange:
    """Model the DocstringChange.

    Represent the docstringchange data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    path: Path


def parse_args() -> argparse.Namespace:
    """Compute parse args.

    Carry out the parse args operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Returns
    -------
    argparse.Namespace
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import parse_args
    >>> result = parse_args()
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path, help="Directory to process.")
    parser.add_argument("--log", required=False, type=Path, help="Log file for changed paths.")
    return parser.parse_args()


def module_name_for(path: Path) -> str:
    """Compute module name for.

    Carry out the module name for operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    path : Path
    path : Path
        Description for ``path``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import module_name_for
    >>> result = module_name_for(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError:
        relative = path

    in_src = path.is_relative_to(SRC_ROOT)
    parts = list(relative.with_suffix("").parts)
    if in_src and parts and parts[0] == "src":
        parts = parts[1:]

    module = ".".join(parts)
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]

    return module


def summarize(name: str, kind: str) -> str:
    """Compute summarize.

    Carry out the summarize operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    name : str
    name : str
        Description for ``name``.
    kind : str
    kind : str
        Description for ``kind``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import summarize
    >>> result = summarize(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    base = _humanize_identifier(name) or "value"
    if kind == "module":
        text = f"Overview of {base}."
    elif kind == "class":
        text = f"Model the {base}."
    elif kind == "function":
        text = f"Compute {base}."
    else:
        text = f"Describe {base}."
    return text if text.endswith(".") else text + "."


def extended_summary(kind: str, name: str, module_name: str, node: ast.AST | None = None) -> str:
    """Compute extended summary.

    Carry out the extended summary operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    kind : str
    kind : str
        Description for ``kind``.
    name : str
    name : str
        Description for ``name``.
    module_name : str
    module_name : str
        Description for ``module_name``.
    node : ast.AST | None
    node : ast.AST | None, optional, default=None
        Description for ``node``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import extended_summary
    >>> result = extended_summary(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    pretty = _humanize_identifier(name)
    if kind == "module":
        module_pretty = _humanize_identifier(module_name.split(".")[-1] if module_name else name)
        if module_pretty:
            return _paragraph(
                f"This module bundles {module_pretty.lower()} logic for the kgfoundry stack.",
                "It groups related helpers so downstream packages can import a single cohesive namespace.",
                "Refer to the functions and classes below for implementation specifics.",
            )
        return _paragraph(
            "Utility module providing KGFoundry helpers.",
            "It exists to keep documentation coverage consistent for internal support code.",
            "Consumers should rely on the documented public functions rather than touching private members.",
        )
    if kind == "class" and isinstance(node, ast.ClassDef):
        if _is_pydantic_model(node):
            return _paragraph(
                "Pydantic model defining the structured payload used across the system.",
                "Validation ensures inputs conform to the declared schema while producing clear error messages.",
                "Use this class when serialising or parsing data for the surrounding feature.",
            )
        if pretty:
            return _paragraph(
                f"Represent the {pretty.lower()} data structure used throughout the project.",
                "The class encapsulates behaviour behind a well-defined interface for collaborating components.",
                "Instances are typically created by factories or runtime orchestrators documented nearby.",
            )
        return _paragraph(
            "Core data structure used within kgfoundry.",
            "It organises related behaviour and provides lifecycle management for the feature domain.",
            "Review the attribute descriptions below for usage guidance.",
        )
    if _is_pydantic_artifact(name):
        return PYDANTIC_ARTIFACT_SUMMARIES.get(name, DEFAULT_PYDANTIC_ARTIFACT_SUMMARY)
    if kind == "function" and name == "__init__":
        return _paragraph(
            "Initialise a new instance with validated parameters.",
            "The constructor prepares internal state and coordinates any setup required by the class.",
            "Subclasses should call ``super().__init__`` to keep validation and defaults intact.",
        )
    if kind == "function" and name in MAGIC_METHOD_EXTENDED_SUMMARIES:
        return MAGIC_METHOD_EXTENDED_SUMMARIES[name]
    if kind == "function" and name in _STANDARD_METHOD_EXTENDED_SUMMARIES:
        return _STANDARD_METHOD_EXTENDED_SUMMARIES[name]
    if kind == "function" and _is_magic(name):
        return DEFAULT_MAGIC_METHOD_FALLBACK
    if kind == "function":
        if pretty:
            return _paragraph(
                f"Carry out the {pretty.lower()} operation for the surrounding component.",
                "Generated documentation highlights how this helper collaborates with neighbouring utilities.",
                "Callers rely on the routine to remain stable across releases.",
            )
        return _paragraph(
            "Perform the requested operation.",
            "The helper abstracts lower-level details so feature code can stay concise.",
            "Consult inline parameter documentation for usage guidance.",
        )
    return _paragraph(
        "Auto-generated reference for project internals.",
        "Documentation generation surfaces private helpers to maintain discoverability.",
        "Most callers should prefer the public APIs documented elsewhere in this module.",
    )


def annotation_to_text(node: ast.AST | None) -> str:
    """Compute annotation to text.

    Carry out the annotation to text operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    node : ast.AST | None
    node : ast.AST | None
        Description for ``node``.
    
    Returns
    -------
    str
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import annotation_to_text
    >>> result = annotation_to_text(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    if node is None:
        return "Any"
    try:
        text = ast.unparse(node)
    except Exception:  # pragma: no cover
        return "Any"
    return _format_annotation_string(text)


def iter_docstring_nodes(tree: ast.Module) -> list[tuple[int, ast.AST, str]]:
    """Compute iter docstring nodes.

    Carry out the iter docstring nodes operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    tree : ast.Module
    tree : ast.Module
        Description for ``tree``.
    
    Returns
    -------
    List[Tuple[int, ast.AST, str]]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import iter_docstring_nodes
    >>> result = iter_docstring_nodes(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    items: list[tuple[int, ast.AST, str]] = [(0, tree, "module")]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            items.append((node.lineno, node, "class"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((node.lineno, node, "function"))
    items.sort(key=lambda item: item[0], reverse=True)
    return items


@dataclass(frozen=True)
class ParameterInfo:
    """Model the ParameterInfo.

    Represent the parameterinfo data structure used throughout the project. The class encapsulates
    behaviour behind a well-defined interface for collaborating components. Instances are typically
    created by factories or runtime orchestrators documented nearby.
    """

    name: str
    annotation: str
    required: bool
    has_default: bool = False
    default_text: str | None = None


def parameters_for(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ParameterInfo]:
    """Compute parameters for.

    Carry out the parameters for operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    node : ast.FunctionDef | ast.AsyncFunctionDef
    node : ast.FunctionDef | ast.AsyncFunctionDef
        Description for ``node``.
    
    Returns
    -------
    List[ParameterInfo]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import parameters_for
    >>> result = parameters_for(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    params: list[ParameterInfo] = []
    args = node.args

    def add(arg: ast.arg, default: ast.AST | None) -> None:
        """Collect metadata for a single parameter."""
        name = arg.arg
        if name in {"self", "cls"}:
            return
        annotation_text = annotation_to_text(arg.annotation)
        display_annotation = annotation_text
        has_default = default is not None
        if has_default and display_annotation.endswith(" | None"):
            display_annotation = display_annotation[: -len(" | None")]
        if has_default:
            cleaned = display_annotation or "Any"
            display_annotation = f"{cleaned} | None"
        default_text: str | None = None
        if default is not None:
            evaluated_default: Any = _DEFAULT_SENTINEL
            try:
                evaluated_default = ast.literal_eval(default)
            except (TypeError, ValueError):
                evaluated_default = _DEFAULT_SENTINEL
            except Exception:  # pragma: no cover - defensive
                evaluated_default = _DEFAULT_SENTINEL
            if evaluated_default is not _DEFAULT_SENTINEL:
                default_text = repr(evaluated_default)
            else:
                default_text = _unparse_or_none(default)
        params.append(
            ParameterInfo(
                name=name,
                annotation=display_annotation or "Any",
                required=not has_default,
                has_default=has_default,
                default_text=default_text,
            )
        )

    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults, strict=True):
        add(arg, default)

    if args.vararg:
        annotation_text = annotation_to_text(args.vararg.annotation)
        params.append(
            ParameterInfo(
                name=f"*{args.vararg.arg}",
                annotation=annotation_text or "Any",
                required=False,
            )
        )

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        add(arg, default)

    if args.kwarg:
        annotation_text = annotation_to_text(args.kwarg.annotation)
        params.append(
            ParameterInfo(
                name=f"**{args.kwarg.arg}",
                annotation=annotation_text or "Any",
                required=False,
            )
        )

    return params


def detect_raises(node: ast.AST) -> list[str]:
    """Compute detect raises.

    Carry out the detect raises operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    node : ast.AST
    node : ast.AST
        Description for ``node``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import detect_raises
    >>> result = detect_raises(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    seen: OrderedDict[str, None] = OrderedDict()

    def _exception_name(exc: ast.AST | None) -> str:
        """Exception name.

        Parameters
        ----------
        exc : ast.AST | None
            Description.

        Returns
        -------
        str
            Description.

        Raises
        ------
        Exception
            Description.

        Examples
        --------
        >>> _exception_name(...)
        """
        if exc is None:
            return "Exception"
        if isinstance(exc, ast.Call):
            func = exc.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return ast.unparse(func)
            return "Exception"  # pragma: no cover - defensive
        if isinstance(exc, ast.Name):
            return exc.id
        if isinstance(exc, ast.Attribute):
            return ast.unparse(exc)
        return "Exception"

    class RaiseCollector(ast.NodeVisitor):
        """Collect ``raise`` statements while respecting scope boundaries."""

        _NESTED_SCOPES = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Lambda,
        )

        def __init__(self, root: ast.AST) -> None:
            """Init  .

            Parameters
            ----------
            root : ast.AST
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
            self._root = root

        def visit(self, current: ast.AST) -> Any:
            """Visit.

            Parameters
            ----------
            current : ast.AST
                Description.

            Returns
            -------
            Any
                Description.

            Raises
            ------
            Exception
                Description.

            Examples
            --------
            >>> visit(...)
            """
            if current is not self._root and isinstance(current, self._NESTED_SCOPES):
                return None
            return super().visit(current)

        def visit_Raise(self, raise_node: ast.Raise) -> None:
            """Visit raise.

            Parameters
            ----------
            raise_node : ast.Raise
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
            >>> visit_Raise(...)
            """
            name = _exception_name(raise_node.exc)
            if name not in seen:
                seen[name] = None
            self.generic_visit(raise_node)

    RaiseCollector(node).visit(node)
    return list(seen.keys())


def build_examples(
    module_name: str, name: str, parameters: list[ParameterInfo], has_return: bool
) -> list[str]:
    """Compute build examples.

    Carry out the build examples operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    module_name : str
    module_name : str
        Description for ``module_name``.
    name : str
    name : str
        Description for ``name``.
    parameters : List[ParameterInfo]
    parameters : List[ParameterInfo]
        Description for ``parameters``.
    has_return : bool
    has_return : bool
        Description for ``has_return``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import build_examples
    >>> result = build_examples(..., ..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    lines: list[str] = ["Examples", "--------"]
    if module_name and not name.startswith("__"):
        lines.append(f">>> from {module_name} import {name}")
    required_args = ["..."] * sum(
        1 for param in parameters if param.required and not param.name.startswith("*")
    )
    variadic_args = [param.name for param in parameters if param.name.startswith("*")]

    call_parts = [*required_args, *variadic_args]
    invocation = f"{name}({', '.join(call_parts)})" if call_parts else f"{name}()"
    if not name.startswith("__"):
        if has_return:
            lines.append(f">>> result = {invocation}")
            lines.append(">>> result  # doctest: +ELLIPSIS")
            lines.append("...")
        else:
            lines.append(f">>> {invocation}  # doctest: +ELLIPSIS")
    return lines


def build_docstring(kind: str, node: ast.AST, module_name: str) -> list[str]:
    """Compute build docstring.

    Carry out the build docstring operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    kind : str
    kind : str
        Description for ``kind``.
    node : ast.AST
    node : ast.AST
        Description for ``node``.
    module_name : str
    module_name : str
        Description for ``module_name``.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import build_docstring
    >>> result = build_docstring(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    if kind == "module":
        module_display = module_name.split(".")[-1] if module_name else "module"
        summary = summarize(module_display, kind)
        extended = extended_summary(kind, module_display, module_name, node)
    else:
        object_name = getattr(node, "name", "value")
        summary = summarize(object_name, kind)
        extended = extended_summary(kind, object_name, module_name, node)

    parameters: list[ParameterInfo] = []
    returns: str | None = None
    raises: list[str] = []
    if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parameters = parameters_for(node)
        return_annotation = annotation_to_text(node.returns)
        if return_annotation not in {"None", "NoReturn"}:
            returns = return_annotation
        raises = detect_raises(node)

    lines: list[str] = ['"""', summary]
    if extended:
        lines.extend(["", extended])

    if kind == "module":
        lines.append('"""')
        return lines

    if kind == "class" and isinstance(node, ast.ClassDef):
        lines.append('"""')
        return lines

    if parameters:
        lines.extend(["", "Parameters", "----------"])
        for parameter in parameters:
            lines.append(f"{parameter.name} : {parameter.annotation}")
            extras: list[str] = []
            if parameter.has_default:
                extras.append("optional")
                if parameter.default_text is not None:
                    extras.append(f"default={parameter.default_text}")
            suffix = f", {', '.join(extras)}" if extras else ""
            lines.append(f"{parameter.name} : {parameter.annotation}{suffix}")
            lines.append(f"    Description for ``{parameter.name}``.")

    if returns:
        lines.extend(["", "Returns", "-------", returns, "    Description of return value."])

    if raises:
        lines.extend(["", "Raises", "------"])
        for exc in raises:
            lines.append(f"{exc}")
            lines.append("    Raised when validation fails.")

    if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = getattr(node, "name", "")
        is_public = bool(name) and not name.startswith("_")
        if is_public:
            example_lines = build_examples(module_name, name, parameters, bool(returns))
            if example_lines:
                lines.extend(["", *example_lines])

    lines.append('"""')
    return lines


def _required_sections(
    kind: str,
    parameters: list[ParameterInfo],
    returns: str | None,
    raises: list[str],
    node_name: str | None,
    include_examples: bool,
) -> set[str]:
    """Return placeholder markers emitted by the fallback generator.

    Parameters
    ----------
    kind : str
        Description for ``kind``.
    parameters : List[ParameterInfo]
        Description for ``parameters``.
    returns : str | None
        Description for ``returns``.
    raises : List[str]
        Description for ``raises``.
    node_name : str | None
        Description for ``node_name``.
    include_examples : bool
        Description for ``include_examples``.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    if kind in {"module", "class"}:
        return set()

    markers: set[str] = set()
    if node_name:
        operation = _humanize_identifier(node_name).lower()
        markers.add(f"Carry out the {operation} operation for the surrounding component.")
        markers.add(
            "Generated documentation highlights how this helper collaborates with neighbouring utilities.",
        )
        markers.add("Callers rely on the routine to remain stable across releases.")
    for parameter in parameters:
        markers.add(f"Description for ``{parameter.name}``.")
    required: set[str] = set()
    if include_examples:
        required.add("Examples")
    if parameters:
        required.add("Parameters")
    if returns:
        markers.add("Description of return value.")
    if raises:
        markers.add("Raised when validation fails.")
    return markers


def docstring_text(node: ast.AST) -> tuple[str | None, ast.Expr | None]:
    """Compute docstring text.

    Carry out the docstring text operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    node : ast.AST
    node : ast.AST
        Description for ``node``.
    
    Returns
    -------
    Tuple[str | None, ast.Expr | None]
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import docstring_text
    >>> result = docstring_text(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    body = getattr(node, "body", [])
    if not body:
        return None, None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value.value, first
    return None, None


def replace(
    doc_expr: ast.Expr | None, lines: list[str], new_lines: list[str], indent: str, insert_at: int
) -> None:
    """Compute replace.

    Carry out the replace operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    doc_expr : ast.Expr | None
    doc_expr : ast.Expr | None
        Description for ``doc_expr``.
    lines : List[str]
    lines : List[str]
        Description for ``lines``.
    new_lines : List[str]
    new_lines : List[str]
        Description for ``new_lines``.
    indent : str
    indent : str
        Description for ``indent``.
    insert_at : int
    insert_at : int
        Description for ``insert_at``.

    Examples
    --------
    >>> from tools.auto_docstrings import replace
    >>> replace(..., ..., ..., ..., ...)  # doctest: +ELLIPSIS
    """
    formatted = [indent + line + "\n" for line in new_lines]
    existing_blank_line = False
    if doc_expr is not None:
        next_line_index = doc_expr.end_lineno or doc_expr.lineno
        if next_line_index < len(lines):
            existing_blank_line = lines[next_line_index].strip() == ""
        start = doc_expr.lineno - 1
        end = doc_expr.end_lineno or doc_expr.lineno
        del lines[start:end]
        lines[start:start] = formatted
        after_index = start + len(formatted)
    else:
        lines[insert_at:insert_at] = formatted
        after_index = insert_at + len(formatted)
    if not existing_blank_line:
        lines.insert(after_index, indent + "\n")


def process_file(path: Path) -> bool:
    """Compute process file.

    Carry out the process file operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    path : Path
    path : Path
        Description for ``path``.
    
    Returns
    -------
    bool
        Description of return value.
    
    Examples
    --------
    >>> from tools.auto_docstrings import process_file
    >>> result = process_file(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    tree = ast.parse(text)
    lines = text.splitlines()
    lines = [line + "\n" for line in lines]
    changed = False

    module_name = module_name_for(path)

    for _, node, kind in iter_docstring_nodes(tree):
        node_name = getattr(node, "name", None)
        if kind != "module" and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if node_name and _is_magic(node_name) and node_name != "__init__":
                if node_name not in MAGIC_METHOD_EXTENDED_SUMMARIES:
                    continue
            if node_name and node_name.startswith("_") and not node_name.startswith("__"):
                continue

        doc, expr = docstring_text(node)
        parameters: list[ParameterInfo] = []
        returns: str | None = None
        raises: list[str] = []
        include_examples = False
        if kind == "function" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = parameters_for(node)
            return_annotation: str = annotation_to_text(node.returns)
            if return_annotation not in {"None", "NoReturn"}:
                returns = return_annotation
            raises = detect_raises(node)
            include_examples = False
            if node_name:
                include_examples = not node_name.startswith("_")

        placeholder_markers = _required_sections(
            kind,
            parameters,
            returns,
            raises,
            node_name,
            include_examples,
        )
        needs_update = doc is None or "TODO" in (doc or "") or "NavMap:" in (doc or "")
        if not needs_update and doc and placeholder_markers:
            if any(marker in doc for marker in placeholder_markers):
                needs_update = True
            else:
                lower_doc = doc.lower()
                lower_markers = tuple(marker.lower() for marker in placeholder_markers)
                if any(marker in lower_doc for marker in lower_markers):
                    needs_update = True
        if not needs_update and doc is not None:
            boilerplate_tokens = (
                "Provide utilities for module.",
                "This module exposes the primary interfaces for the package.",
                "See Also\n--------\n",
                "Attributes\n----------\nNone",
                "Method description.",
                "Provide usage considerations, constraints, or complexity notes.",
            )
            if any(token in doc for token in boilerplate_tokens):
                needs_update = True
        if kind == "module" and doc is None:
            continue
        if not needs_update:
            continue

        new_lines = build_docstring(kind, node, module_name)
        generated_body = "\n".join(new_lines[1:-1]) if len(new_lines) >= 2 else ""
        existing_body = inspect.cleandoc(doc) if doc is not None else ""
        if doc is not None and existing_body == generated_body:
            continue

        if kind == "module":
            indent = ""
            insert_at = 1 if lines and lines[0].startswith("#!") else 0
            replace(expr, lines, new_lines, indent, insert_at)
        else:
            if not isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):
                continue
            indent = " " * (node.col_offset + 4)
            body: list[ast.stmt] = node.body
            insert_at = body[0].lineno - 1 if body else node.lineno
            replace(expr, lines, new_lines, indent, insert_at)
        changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Examples
    --------
    >>> from tools.auto_docstrings import main
    >>> main()  # doctest: +ELLIPSIS
    """
    args = parse_args()
    target = args.target.resolve()
    changed: list[DocstringChange] = []

    for file_path in sorted(target.rglob("*.py")):
        if process_file(file_path):
            changed.append(DocstringChange(file_path))

    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a", encoding="utf-8") as handle:
            for item in changed:
                handle.write(f"{item.path}\n")


if __name__ == "__main__":
    main()
