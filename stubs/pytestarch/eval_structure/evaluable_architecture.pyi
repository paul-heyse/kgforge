from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pytestarch.pytestarch import ModuleDependencies

class ModuleFilter: ...

class EvaluableArchitecture:  # pragma: no cover - stub only
    modules: tuple[str, ...]

    def get_dependencies(
        self,
        sources: Iterable[ModuleFilter],
        targets: Iterable[ModuleFilter],
    ) -> ModuleDependencies: ...

class LayerMapping(Mapping[str, Sequence[ModuleFilter]]): ...
