"""Sample transformer plugin that normalises NumPy parameter descriptions."""

from __future__ import annotations

from dataclasses import replace

from tools.docstring_builder.plugins import PluginContext, TransformerPlugin
from tools.docstring_builder.schema import ParameterDoc, ReturnDoc
from tools.docstring_builder.semantics import SemanticResult


class NormalizeNumpyParamsPlugin(TransformerPlugin):
    """Ensure parameter descriptions follow a consistent style."""

    name = "normalize_numpy_params"

    def run(self, context: PluginContext, result: SemanticResult) -> SemanticResult:
        schema = result.schema
        parameters = [self._normalize_parameter(parameter) for parameter in schema.parameters]
        returns = [self._normalize_return(entry) for entry in schema.returns]
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

