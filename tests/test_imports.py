def test_imports() -> None:
    import kgfoundry.kgfoundry_common.models as m

    assert hasattr(m, "Doc")
