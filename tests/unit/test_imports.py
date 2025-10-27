import importlib


def test_imports():
    for mod in [
        "kgforge.kgforge_common.models",
        "kgforge.download.harvester",
        "kgforge.search_api.app",
        "kgforge.registry.migrate",
        "kgforge.registry.helper",
        "kgforge.orchestration.flows",
        "kgforge.orchestration.fixture_flow",
    ]:
        importlib.import_module(mod)
