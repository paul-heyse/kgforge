"""Role-based access control helpers for Agent Catalog hosted mode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


class Role(StrEnum):
    """Supported access roles for hosted mode.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    *values : inspect._empty
        Describe ``values``.

    Returns
    -------
    inspect._empty
        Describe return value.
    """

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    ADMIN = "admin"


@dataclass(slots=True)
class AccessController:
    """Authorize catalog operations based on the configured role.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    role : Role
        Describe ``role``.
    enabled : bool, optional
        Describe ``enabled``.
        Defaults to ``False``.
    """

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
        """Raise :class:`PermissionError` if ``method`` is not permitted.

        <!-- auto:docstring-builder v1 -->

        Parameters
        ----------
        method : str
            Describe ``method``.

        Raises
        ------
        PermissionError
            If the method is not permitted for the current role.
        """
        if not self.enabled:
            return
        allowed = self._PERMISSIONS.get(self.role, frozenset())
        if method not in allowed:
            message = f"Role '{self.role.value}' may not invoke {method}"
            raise PermissionError(message)


__all__ = ["AccessController", "Role"]
