from kgfoundry_common.config import AppSettings


def test_import() -> None:
    settings = AppSettings()
    assert settings.log_level == "INFO"
