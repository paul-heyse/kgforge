"""Validate documentation type resolution helpers and configuration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tools.auto_docstrings import QUALIFIED_NAME_OVERRIDES, _normalize_qualified_name

CONF_PATH = Path(__file__).resolve().parents[2] / "docs" / "conf.py"


def _load_conf_symbol(name: str) -> object:
    """Return the literal value assigned to ``name`` inside ``docs/conf.py``."""
    module = ast.parse(CONF_PATH.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == name
                and node.value is not None
            ):
                return ast.literal_eval(node.value)
    raise KeyError(name)


@pytest.fixture(name="overrides")
def overrides_fixture() -> dict[str, str]:
    """Return the qualified name override mapping used during documentation generation."""
    return QUALIFIED_NAME_OVERRIDES


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("numpy.float16", "numpy.float16"),
        ("numpy.float32", "numpy.float32"),
        ("numpy.float64", "numpy.float64"),
        ("numpy.int8", "numpy.int8"),
        ("numpy.int16", "numpy.int16"),
        ("numpy.int32", "numpy.int32"),
        ("numpy.int64", "numpy.int64"),
        ("numpy.uint8", "numpy.uint8"),
        ("numpy.uint16", "numpy.uint16"),
        ("numpy.uint32", "numpy.uint32"),
        ("numpy.uint64", "numpy.uint64"),
        ("np.float16", "numpy.float16"),
        ("np.float32", "numpy.float32"),
        ("np.float64", "numpy.float64"),
        ("np.int8", "numpy.int8"),
        ("np.int16", "numpy.int16"),
        ("np.int32", "numpy.int32"),
        ("np.int64", "numpy.int64"),
        ("np.uint8", "numpy.uint8"),
        ("np.uint16", "numpy.uint16"),
        ("np.uint32", "numpy.uint32"),
        ("np.uint64", "numpy.uint64"),
    ],
)
def test_numpy_scalar_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure NumPy scalar aliases resolve to canonical types."""
    assert overrides[alias] == canonical
    assert _normalize_qualified_name(alias) == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("ArrayLike", "numpy.typing.ArrayLike"),
        ("NDArray", "numpy.typing.NDArray"),
        ("numpy.dtype", "numpy.dtype"),
        ("numpy.typing.ArrayLike", "numpy.typing.ArrayLike"),
        ("numpy.typing.NDArray", "numpy.typing.NDArray"),
        ("np.dtype", "numpy.dtype"),
        ("np.ndarray", "numpy.ndarray"),
        ("np.typing.NDArray", "numpy.typing.NDArray"),
    ],
)
def test_numpy_typing_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure ndarray and typing helpers resolve to canonical targets."""
    assert overrides[alias] == canonical
    assert _normalize_qualified_name(alias) == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("pyarrow.Array", "pyarrow.Array"),
        ("pyarrow.DataType", "pyarrow.DataType"),
        ("pyarrow.Field", "pyarrow.Field"),
        ("pyarrow.Int64Type", "pyarrow.Int64Type"),
        ("pyarrow.RecordBatch", "pyarrow.RecordBatch"),
        ("pyarrow.Schema", "pyarrow.Schema"),
        ("pyarrow.schema", "pyarrow.schema"),
        ("pyarrow.StringType", "pyarrow.StringType"),
        ("pyarrow.Table", "pyarrow.Table"),
        ("pyarrow.TimestampType", "pyarrow.TimestampType"),
    ],
)
def test_pyarrow_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure PyArrow core types resolve to canonical targets."""
    assert overrides[alias] == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("pydantic.AliasChoices", "pydantic.AliasChoices"),
        ("pydantic.ConfigDict", "pydantic.ConfigDict"),
        ("pydantic.Field", "pydantic.Field"),
        ("pydantic.TypeAdapter", "pydantic.TypeAdapter"),
        ("pydantic.ValidationError", "pydantic.ValidationError"),
        ("pydantic.field_validator", "pydantic.field_validator"),
        ("pydantic.fields.Field", "pydantic.fields.Field"),
        ("pydantic.model_validator", "pydantic.model_validator"),
    ],
)
def test_pydantic_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure Pydantic helpers are present in override mapping."""
    assert overrides[alias] == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("typing_extensions.Annotated", "typing_extensions.Annotated"),
        ("typing_extensions.NotRequired", "typing_extensions.NotRequired"),
        ("typing_extensions.Self", "typing_extensions.Self"),
        ("typing_extensions.TypeAlias", "typing_extensions.TypeAlias"),
        ("typing_extensions.TypedDict", "typing_extensions.TypedDict"),
    ],
)
def test_typing_extensions_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure typing_extensions utilities resolve via overrides."""
    assert overrides[alias] == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("collections.Counter", "collections.Counter"),
        ("collections.defaultdict", "collections.defaultdict"),
        ("collections.deque", "collections.deque"),
        ("collections.OrderedDict", "collections.OrderedDict"),
        ("datetime.datetime", "datetime.datetime"),
        ("datetime.timedelta", "datetime.timedelta"),
        ("pathlib.Path", "pathlib.Path"),
        ("uuid.UUID", "uuid.UUID"),
    ],
)
def test_standard_library_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure frequently documented stdlib types resolve."""
    assert overrides[alias] == canonical


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("FloatArray", "src.vectorstore_faiss.gpu.FloatArray"),
        ("IntArray", "src.vectorstore_faiss.gpu.IntArray"),
        ("StrArray", "src.vectorstore_faiss.gpu.StrArray"),
        ("VecArray", "src.search_api.faiss_adapter.VecArray"),
        ("Doc", "src.kgfoundry_common.models.Doc"),
        ("Chunk", "src.kgfoundry_common.models.Chunk"),
        ("Concept", "src.ontology.catalog.Concept"),
    ],
)
def test_custom_project_overrides(overrides: dict[str, str], alias: str, canonical: str) -> None:
    """Ensure project-specific aliases remain canonicalised."""
    assert overrides[alias] == canonical


def test_generic_annotations_collapse_to_base() -> None:
    """Normalisation should drop generic parameters for unresolved types."""
    result = _normalize_qualified_name("numpy.typing.NDArray[numpy.float32]")
    assert result == "numpy.typing.NDArray"


def test_exception_overrides_are_canonical(overrides: dict[str, str]) -> None:
    """Duplicate exception targets should point to the errors module."""
    assert overrides["DownloadError"] == "src.kgfoundry_common.errors.DownloadError"
    assert overrides["UnsupportedMIMEError"] == "src.kgfoundry_common.errors.UnsupportedMIMEError"


def test_override_count_exceeds_target(overrides: dict[str, str]) -> None:
    """The override catalogue should exceed the 100-entry target."""
    assert len(overrides) >= 100


def test_autoapi_excludes_legacy_exceptions() -> None:
    """AutoAPI should ignore the deprecated exceptions module."""
    ignore_patterns = _load_conf_symbol("autoapi_ignore")
    assert "*/kgfoundry_common/exceptions.py" in ignore_patterns


def test_intersphinx_mappings_cover_required_projects() -> None:
    """Verify intersphinx configuration includes scientific and http stacks."""
    mapping = _load_conf_symbol("intersphinx_mapping")
    for project in ("scipy", "pandas", "httpx", "pytest"):
        assert project in mapping


def test_extlinks_provide_fallback_type_links() -> None:
    """Ensure fallback external links exist for unmapped type references."""
    links = _load_conf_symbol("extlinks")
    assert "numpy-type" in links
    assert "pyarrow-type" in links


def test_extlinks_render_sample_types() -> None:
    """Verify fallback link templates interpolate sample type names correctly."""
    links = _load_conf_symbol("extlinks")
    numpy_template, numpy_label = links["numpy-type"]
    pyarrow_template, pyarrow_label = links["pyarrow-type"]
    assert "numpy.float32" in numpy_template % "numpy.float32"
    assert numpy_label % "numpy.float32" == "numpy.float32"
    assert "pyarrow.Table" in pyarrow_template % "pyarrow.Table"
    assert pyarrow_label % "pyarrow.Table" == "pyarrow.Table"
