def test_imports():
    import kgfoundry.kgfoundry_common.models as m

    assert hasattr(m, "Doc")
