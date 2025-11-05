"""Role-based access control helpers for Agent Catalog hosted mode."""
# [nav:section public-api]

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from kgfoundry_common.navmap_loader import load_nav_metadata

if TYPE_CHECKING:
    from collections.abc import Mapping


# [nav:anchor Role]
class Role(StrEnum):
    """Supported access roles for hosted mode.

    Enumeration of access roles used for role-based access control (RBAC)
    in hosted Agent Catalog deployments. Roles define different permission
    levels for catalog operations.

    Roles
    -----
    VIEWER
        Read-only access. Can search and view catalog contents but cannot
        modify or perform advanced operations.
    CONTRIBUTOR
        Read-write access. Can search, view, and perform analysis operations
        like find_callers, find_callees, change_impact, and suggest_tests.
    ADMIN
        Full access. Can perform all operations including open_anchor,
        session management, and shutdown operations.

    Examples
    --------
    >>> role = Role.VIEWER
    >>> assert role == "viewer"
    >>> assert isinstance(role, Role)
    """

    VIEWER = "viewer"
    CONTRIBUTOR = "contributor"
    ADMIN = "admin"


@dataclass(slots=True)
# [nav:anchor AccessController]
class AccessController:
    """Authorize catalog operations based on the configured role.

    Provides role-based access control (RBAC) for Agent Catalog operations.
    Each role has a predefined set of permitted methods. When enabled, the
    controller raises PermissionError for unauthorized method calls.

    Parameters
    ----------
    role : Role
        Access role assigned to this controller.
    enabled : bool, optional
        Whether RBAC is enabled. If False, all method calls are allowed.
        Defaults to False.

    Attributes
    ----------
    role : Role
        Access role assigned to this controller.
    enabled : bool
        Whether RBAC is enabled.
    _PERMISSIONS : ClassVar[Mapping[Role, frozenset[str]]]
        Mapping of roles to their permitted method names.

    Examples
    --------
    >>> controller = AccessController(Role.VIEWER, enabled=True)
    >>> controller.authorize("catalog.search")  # OK
    >>> controller.authorize("catalog.open_anchor")  # Raises PermissionError
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
        """Raise PermissionError if method is not permitted.

        Checks if the given method is permitted for the current role.
        If RBAC is disabled or the method is permitted, returns normally.
        Otherwise, raises PermissionError with a descriptive message.

        Parameters
        ----------
        method : str
            Method name to authorize (e.g., "catalog.search", "catalog.open_anchor").

        Raises
        ------
        PermissionError
            If the method is not permitted for the current role.

        Notes
        -----
        If RBAC is disabled (enabled=False), this method always returns
        without raising an error.

        Examples
        --------
        >>> controller = AccessController(Role.VIEWER, enabled=True)
        >>> controller.authorize("catalog.search")  # OK
        >>> controller.authorize("catalog.open_anchor")  # Raises PermissionError
        """
        if not self.enabled:
            return
        allowed = self._PERMISSIONS.get(self.role, frozenset())
        if method not in allowed:
            message = f"Role '{self.role.value}' may not invoke {method}"
            raise PermissionError(message)


__all__ = [
    "AccessController",
    "Role",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))
