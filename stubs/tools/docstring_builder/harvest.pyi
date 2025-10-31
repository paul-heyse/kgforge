from collections.abc import Iterable
from typing import Any

HarvestResult = Any
SymbolHarvest = Any
ParameterHarvest = Any

def harvest_file(path: Any, config: Any) -> HarvestResult: ...
def iter_target_files(paths: Iterable[Any], config: Any) -> Iterable[Any]: ...
