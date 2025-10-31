"""Plugin that normalizes parameter descriptions for NumPy style docstrings."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from tools.docstring_builder.plugins.base import PluginContext, PluginStage
from tools.docstring_builder.schema import ParameterDoc, ReturnDoc
from tools.docstring_builder.semantics import SemanticResult


class NormalizeNumpyParamsPlugin:
    """Ensure parameter descriptions follow a consistent style."""

    name: ClassVar[str] = "normalize_numpy_params"
    stage: ClassVar[PluginStage] = "transformer"

    @staticmethod
    def on_start(context: PluginContext) -> None:
        """Reset plugin state before processing begins (no-op)."""
        del context

    @staticmethod
    def on_finish(context: PluginContext) -> None:
        """Clean up plugin state after processing completes (no-op)."""
        del context

    @staticmethod
    def apply(context: PluginContext, result: SemanticResult) -> SemanticResult:
        """Normalise parameter and return descriptions for ``result``."""
        del context
        schema = result.schema
        parameters = [
            NormalizeNumpyParamsPlugin._normalize_parameter(parameter)
            for parameter in schema.parameters
        ]
        returns = [NormalizeNumpyParamsPlugin._normalize_return(entry) for entry in schema.returns]
        if parameters == schema.parameters and returns == schema.returns:
            return result
        updated_schema = replace(schema, parameters=parameters, returns=returns)
        return replace(result, schema=updated_schema)

    @staticmethod
    def _normalize_parameter(parameter: ParameterDoc) -> ParameterDoc:
        description = parameter.description.strip()
        if not description or description.lower().startswith("todo"):
            description = f"Describe `{parameter.name}`."  # pragma: no cover - exercised via tests
        if not description.endswith("."):
            description = f"{description}."
        if description and description[0].islower():
            description = description[0].upper() + description[1:]
        return replace(parameter, description=description)

    @staticmethod
    def _normalize_return(entry: ReturnDoc) -> ReturnDoc:
        description = entry.description.strip()
        if not description or description.lower().startswith("todo"):
            description = "Describe return value."
        if not description.endswith("."):
            description = f"{description}."
        if description and description[0].islower():
            description = description[0].upper() + description[1:]
        return replace(entry, description=description)
