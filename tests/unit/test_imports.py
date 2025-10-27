import importlib


def test_imports():
    for mod in [
        "kgfoundry.kgfoundry_common.models",
        "kgfoundry.download.harvester",
        "kgfoundry.search_api.app",
        "kgfoundry.registry.migrate",
        "kgfoundry.registry.helper",
        "kgfoundry.orchestration.flows",
        "kgfoundry.orchestration.fixture_flow",
    ]:
        importlib.import_module(mod)
