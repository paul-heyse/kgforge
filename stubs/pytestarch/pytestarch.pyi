from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence


class Identifier:  # pragma: no cover - stub only
    identifier: str


DependencyEdge = tuple[Identifier, Identifier]


class ModuleDependencies(Mapping[str, Sequence[DependencyEdge]]):  # pragma: no cover - stub only
    ...


class EvaluableArchitecture:  # pragma: no cover - stub only
    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[object],
        targets: Iterable[object],
    ) -> ModuleDependencies: ...
