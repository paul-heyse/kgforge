from kgfoundry_common.config import AppSettings


def test_import() -> None:
    """Test that kgfoundry_common.config can be imported and instantiated."""
    settings = AppSettings()
    assert settings.log_level == "INFO"
