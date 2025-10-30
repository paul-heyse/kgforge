import jsonschema
import pytest
from tools.docstring_builder.ir import (
    IRDocstring,
    IRParameter,
    IRReturn,
    generate_schema,
    serialize_ir,
    validate_ir,
)


def _sample_ir() -> IRDocstring:
    return IRDocstring(
        symbol_id="pkg.module:func",
        module="pkg.module",
        kind="function",
        source_path="src/pkg/module.py",
        lineno=10,
        summary="Do something.",
        parameters=[
            IRParameter(
                name="value",
                annotation="int",
                optional=False,
                default=None,
                description="Input value.",
                kind="positional_or_keyword",
                display_name=None,
            )
        ],
        returns=[
            IRReturn(annotation="int", description="Result.", kind="returns"),
        ],
        notes=["Examples go here."],
    )


def test_ir_schema_validation() -> None:
    ir = _sample_ir()
    validate_ir(ir)
    data = serialize_ir(ir)
    schema = generate_schema()
    jsonschema.validate(data, schema)


def test_ir_schema_rejects_invalid_kind() -> None:
    ir = _sample_ir()
    ir.kind = "module"
    with pytest.raises(ValueError):
        validate_ir(ir)
    data = serialize_ir(ir)
    schema = generate_schema()
    data["kind"] = "module"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)
