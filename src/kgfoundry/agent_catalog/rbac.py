"""Role-based access control helpers for Agent Catalog hosted mode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Mapping


class Role(str, Enum):
    """Supported access roles for hosted mode."""

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    ADMIN = "admin"


@dataclass(slots=True)
class AccessController:
    """Authorize catalog operations based on the configured role."""

    role: Role
    enabled: bool = False

    _PERMISSIONS: ClassVar[Mapping[Role, frozenset[str]]] = {
        Role.VIEWER: frozenset(
            {
                "initialize",
                "catalog.capabilities",
                "catalog.symbol",
                "catalog.search",
                "catalog.list_modules",
            }
        ),
        Role.CONTRIBUTOR: frozenset(
            {
                "initialize",
                "catalog.capabilities",
                "catalog.symbol",
                "catalog.search",
                "catalog.list_modules",
                "catalog.find_callers",
                "catalog.find_callees",
                "catalog.change_impact",
                "catalog.suggest_tests",
            }
        ),
        Role.ADMIN: frozenset(
            {
                "initialize",
                "catalog.capabilities",
                "catalog.symbol",
                "catalog.search",
                "catalog.list_modules",
                "catalog.find_callers",
                "catalog.find_callees",
                "catalog.change_impact",
                "catalog.suggest_tests",
                "catalog.open_anchor",
                "session.shutdown",
                "session.exit",
            }
        ),
    }

    def authorize(self, method: str) -> None:
        """Raise :class:`PermissionError` if ``method`` is not permitted."""

        if not self.enabled:
            return
        allowed = self._PERMISSIONS.get(self.role, frozenset())
        if method not in allowed:
            message = f"Role '{self.role.value}' may not invoke {method}"
            raise PermissionError(message)


__all__ = ["AccessController", "Role"]
