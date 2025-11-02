from __future__ import annotations

from collections.abc import Iterable, Mapping

class Identifier:  # pragma: no cover - stub only
    identifier: str

class DependencyEdge:  # pragma: no cover - stub only
    importer: Identifier
    importee: Identifier

class ModuleDependencies(Mapping[str, list[DependencyEdge]]):  # pragma: no cover - stub only
    ...

class EvaluableArchitecture:  # pragma: no cover - stub only
    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[object],
        targets: Iterable[object],
    ) -> ModuleDependencies: ...
