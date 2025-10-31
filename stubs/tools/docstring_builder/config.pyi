from typing import Any

BuilderConfig = Any
ConfigSelection = Any
PackageSettings = Any

DEFAULT_CONFIG_PATH: Any
DEFAULT_MARKER: str

def load_config_with_selection(path: Any | None = ...) -> tuple[Any, ConfigSelection]: ...
