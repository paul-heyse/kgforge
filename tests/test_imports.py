def test_imports():
    import kgforge.kgforge_common.models as m

    assert hasattr(m, "Doc")
