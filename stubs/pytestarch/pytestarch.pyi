from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence

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
