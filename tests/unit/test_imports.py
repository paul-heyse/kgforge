import importlib
from collections.abc import Iterable


def test_imports() -> None:
    modules: Iterable[str] = [
        "kgfoundry.kgfoundry_common.models",
        "kgfoundry.download.harvester",
        "kgfoundry.search_api.app",
        "kgfoundry.registry.migrate",
        "kgfoundry.registry.helper",
        "kgfoundry.orchestration.flows",
        "kgfoundry.orchestration.fixture_flow",
    ]
    for mod in modules:
        importlib.import_module(mod)
