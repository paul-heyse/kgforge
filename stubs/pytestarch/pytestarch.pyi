from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from types import ModuleType

DEFAULT_EXCLUSIONS: tuple[str, ...]

class Identifier:  # pragma: no cover - stub only
    identifier: str

DependencyEdge = tuple[Identifier, Identifier]

class ModuleDependencies(Mapping[str, Sequence[DependencyEdge]]):  # pragma: no cover - stub only
    def __getitem__(self, __k: str) -> Sequence[DependencyEdge]: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

class EvaluableArchitecture:  # pragma: no cover - stub only
    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[object],
        targets: Iterable[object],
    ) -> ModuleDependencies: ...

def get_evaluable_architecture(
    root_path: str,
    module_path: str,
    exclusions: tuple[str, ...] = ...,
    exclude_external_libraries: bool = ...,
    level_limit: int | None = ...,
    regex_exclusions: tuple[str, ...] | None = ...,
    external_exclusions: tuple[str, ...] | None = ...,
    regex_external_exclusions: tuple[str, ...] | None = ...,
) -> EvaluableArchitecture: ...
def get_evaluable_architecture_for_module_objects(
    root_module: ModuleType,
    module: ModuleType,
    exclusions: tuple[str, ...] = ...,
    exclude_external_libraries: bool = ...,
    level_limit: int | None = ...,
    regex_exclusions: tuple[str, ...] | None = ...,
    external_exclusions: tuple[str, ...] | None = ...,
    regex_external_exclusions: tuple[str, ...] | None = ...,
) -> EvaluableArchitecture: ...
