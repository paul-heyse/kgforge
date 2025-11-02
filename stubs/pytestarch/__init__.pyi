from __future__ import annotations

from collections.abc import Iterable

from pytestarch.pytestarch import EvaluableArchitecture

class _LayerDefinition:  # pragma: no cover - stub only
    def containing_modules(self, modules: Iterable[str]) -> _LayerDefinition: ...

class LayeredArchitecture:  # pragma: no cover - stub only
    def layer(self, name: str) -> _LayerDefinition: ...

def get_evaluable_architecture(
    project_root: str,
    package_root: str,
) -> EvaluableArchitecture: ...
