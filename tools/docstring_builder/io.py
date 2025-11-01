"""Filesystem and selection helpers for the docstring builder."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from tools._shared.logging import get_logger
from tools._shared.proc import ToolExecutionError, run_tool
from tools.docstring_builder.config import BuilderConfig
from tools.docstring_builder.harvest import iter_target_files
from tools.docstring_builder.paths import REPO_ROOT

LOGGER = get_logger(__name__)

DEFAULT_IGNORE_PATTERNS: list[str] = [
    "tests/e2e/**",
    "tests/mock_servers/**",
    "tests/tools/**",
    "docs/_scripts/**",
    "docs/conf.py",
    "src/__init__.py",
]


class InvalidPathError(ValueError):
    """Raised when a user-supplied path falls outside the allowed workspace."""


def resolve_ignore_patterns(config: BuilderConfig) -> list[str]:
    """Combine default ignore patterns with configuration overrides."""
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    for pattern in config.ignore:
        if pattern not in patterns:
            patterns.append(pattern)
    return patterns


def normalize_input_path(raw: str, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve ``raw`` into an absolute Python source path within ``repo_root``."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        message = f"Path '{raw}' does not exist"
        raise InvalidPathError(message) from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        message = f"Path '{raw}' escapes the repository root"
        raise InvalidPathError(message) from exc
    if not resolved.is_file() or resolved.suffix != ".py":
        message = f"Path '{raw}' must reference a Python source file"
        raise InvalidPathError(message)
    return resolved


def module_to_path(module: str, *, repo_root: Path = REPO_ROOT) -> Path | None:
    """Best-effort mapping from dotted module name to a filesystem path."""
    if not module:
        return None
    parts = module.split(".")
    relative = Path("src", *parts)
    file_candidate = repo_root / relative
    if file_candidate.suffix:
        return file_candidate
    file_path = file_candidate.with_suffix(".py")
    if file_path.exists():
        return file_path
    package_init = file_candidate / "__init__.py"
    if package_init.exists():
        return package_init
    return file_path


def module_name_from_path(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    """Derive a dotted module name from ``path`` relative to ``repo_root``."""
    rel = path.relative_to(repo_root)
    parts = rel.with_suffix("").parts
    if parts and parts[0] in {"src", "tools", "docs"}:
        parts = parts[1:]
    return ".".join(parts)


def dependents_for(path: Path) -> set[Path]:
    """Return potential dependent files that should be processed together."""
    dependents: set[Path] = set()
    if path.name == "__init__.py":
        for candidate in path.parent.glob("*.py"):
            if candidate != path and candidate.is_file():
                dependents.add(candidate.resolve())
    else:
        init_file = (path.parent / "__init__.py").resolve()
        if init_file.exists():
            dependents.add(init_file)
    return dependents


def matches_patterns(path: Path, patterns: Iterable[str], *, repo_root: Path = REPO_ROOT) -> bool:
    """Return ``True`` when ``path`` matches any ``patterns`` relative to ``repo_root``."""
    try:
        rel = path.relative_to(repo_root)
    except ValueError:  # pragma: no cover - defensive guard
        rel = path
    return any(rel.match(pattern) for pattern in patterns)


def should_ignore(path: Path, config: BuilderConfig, *, repo_root: Path = REPO_ROOT) -> bool:
    """Return ``True`` when ``path`` should be ignored according to ``config``."""
    rel = path.relative_to(repo_root)
    patterns = resolve_ignore_patterns(config)
    for pattern in patterns:
        if rel.match(pattern):
            LOGGER.debug("Skipping %s because it matches ignore pattern %s", rel, pattern)
            return True
    return False


def changed_files_since(revision: str, *, repo_root: Path = REPO_ROOT) -> set[str]:
    """Return the set of file paths changed since ``revision``."""
    cmd = ["git", "-C", str(repo_root), "diff", "--name-only", revision, "HEAD", "--"]
    try:
        result = run_tool(cmd, timeout=20.0)
    except ToolExecutionError as exc:  # pragma: no cover - git missing
        LOGGER.warning("git diff failed: %s", exc)
        return set()
    if result.returncode != 0:
        LOGGER.warning("git diff failed: %s", result.stderr.strip())
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def default_since_revision(*, repo_root: Path = REPO_ROOT) -> str | None:
    """Return a sensible default revision for ``--changed-only`` runs."""
    candidates = [
        ["git", "-C", str(repo_root), "merge-base", "HEAD", "origin/main"],
        ["git", "-C", str(repo_root), "rev-parse", "HEAD~1"],
    ]
    for cmd in candidates:
        try:
            result = run_tool(cmd, timeout=10.0)
        except ToolExecutionError:
            continue
        if result.returncode == 0:
            revision = result.stdout.strip()
            if revision:
                return revision
    return None


def select_files(  # noqa: C901, PLR0913
    config: BuilderConfig,
    *,
    module: str | None = None,
    since: str | None = None,
    changed_only: bool = False,
    explicit_paths: Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[Path]:
    """Return the set of candidate files based on CLI-style selection options."""
    if explicit_paths:
        return [normalize_input_path(raw, repo_root=repo_root) for raw in explicit_paths]

    files: list[Path] = []
    for path in iter_target_files(config, repo_root):
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError:  # pragma: no cover - stale glob entry
            continue
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            LOGGER.warning("Ignoring path outside repository: %s", resolved)
            continue
        if should_ignore(resolved, config, repo_root=repo_root):
            continue
        files.append(resolved)

    if module:
        files = [
            candidate
            for candidate in files
            if module_name_from_path(candidate, repo_root=repo_root).startswith(module)
        ]
    if since:
        changed = set(changed_files_since(since, repo_root=repo_root))
        files = [path for path in files if str(path.relative_to(repo_root)) in changed]
    candidates = [
        file_path
        for file_path in files
        if not should_ignore(file_path, config, repo_root=repo_root)
    ]
    if changed_only or since:
        expanded: dict[Path, None] = {candidate.resolve(): None for candidate in candidates}
        for candidate in list(expanded.keys()):
            for dependent in dependents_for(candidate):
                if not should_ignore(dependent, config, repo_root=repo_root):
                    expanded.setdefault(dependent, None)
        candidates = sorted(expanded.keys())
    return candidates


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_baseline_version(baseline: str, path: Path, *, repo_root: Path = REPO_ROOT) -> str | None:
    """Return the file contents for ``path`` from ``baseline`` when available."""
    if not baseline:
        return None
    candidate = Path(baseline)
    relative = path.relative_to(repo_root)
    if candidate.exists():
        base_path = candidate / relative if candidate.is_dir() else candidate
        try:
            return base_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
    command = [
        "git",
        "-C",
        str(repo_root),
        "show",
        f"{baseline}:{relative.as_posix()}",
    ]
    try:
        result = run_tool(command, timeout=10.0)
    except ToolExecutionError as exc:  # pragma: no cover - git missing
        LOGGER.debug(
            "Unable to read %s from baseline %s: %s",
            relative,
            baseline,
            exc,
        )
        return None
    if result.returncode != 0:
        LOGGER.debug(
            "Unable to read %s from baseline %s: %s",
            relative,
            baseline,
            result.stderr,
        )
        return None
    return result.stdout


__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "InvalidPathError",
    "changed_files_since",
    "default_since_revision",
    "dependents_for",
    "hash_file",
    "matches_patterns",
    "module_name_from_path",
    "module_to_path",
    "normalize_input_path",
    "read_baseline_version",
    "resolve_ignore_patterns",
    "select_files",
    "should_ignore",
]
