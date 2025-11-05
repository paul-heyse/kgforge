"""Filesystem utilities using pathlib for safe, typed operations.

This module provides helpers for directory management, file I/O, and path
operations using `pathlib.Path` exclusively. All functions are fully typed
and documented to replace legacy `os.path` usage.

Examples
--------
>>> from pathlib import Path
>>> from kgfoundry_common.fs import ensure_dir, read_text, write_text
>>> base = Path("/tmp/test")
>>> ensure_dir(base / "subdir")
>>> write_text(base / "file.txt", "content")
>>> content = read_text(base / "file.txt")
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Literal

from kgfoundry_common.logging import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Path, *, exist_ok: bool = True) -> Path:
    """Create directory if it does not exist, including parent directories.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Directory path to create.
    exist_ok : bool, optional
        If True, do not raise if directory already exists.
        Defaults to True.
        Defaults to ``True``.

    Returns
    -------
    Path
        The created or existing directory path.

    Raises
    ------
    PermissionError
        If the filesystem denies creation.
    FileExistsError
        If `exist_ok=False` and the path exists as a file.

    Examples
    --------
    >>> from pathlib import Path
    >>> ensure_dir(Path("/tmp/data/subdir"))
    Path('/tmp/data/subdir')
    """
    try:
        path.mkdir(parents=True, exist_ok=exist_ok)
    except (PermissionError, FileExistsError):
        raise
    return path


def safe_join(base: Path, *parts: str | Path) -> Path:
    """Join path components safely, preventing directory traversal.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    base : Path
        Base directory path (must be absolute).
    *parts : str | Path
        Relative path components to join.

    Returns
    -------
    Path
        Resolved path within the base directory.

    Raises
    ------
    ValueError
        If the resolved path escapes the base directory.

    Examples
    --------
    >>> from pathlib import Path
    >>> base = Path("/safe/base")
    >>> safe_join(base, "file.txt")
    Path('/safe/base/file.txt')
    >>> safe_join(base, "..", "etc", "passwd")  # doctest: +SKIP
    ValueError: Path escapes base directory
    """
    if not base.is_absolute():
        msg = f"Base path must be absolute: {base}"
        raise ValueError(msg)
    resolved = (base / Path(*parts)).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        msg = f"Path escapes base directory: {resolved}"
        raise ValueError(msg) from exc
    return resolved


def read_text(path: Path, encoding: str = "utf-8") -> str:
    """Read text file contents with explicit encoding.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        File path to read.
    encoding : str, optional
        Text encoding to use. Defaults to "utf-8".
        Defaults to ``'utf-8'``.

    Returns
    -------
    str
        File contents as text.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    UnicodeDecodeError
        If the file cannot be decoded with the specified encoding.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_text(Path("/tmp/test.txt"), "hello")
    >>> read_text(Path("/tmp/test.txt"))
    'hello'
    """
    try:
        return path.read_text(encoding=encoding)
    except (FileNotFoundError, UnicodeDecodeError):
        raise


def write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Write text data to a file, creating parent directories if needed.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        File path to write.
    data : str
        Text content to write.
    encoding : str, optional
        Text encoding to use. Defaults to "utf-8".
        Defaults to ``'utf-8'``.

    Raises
    ------
    PermissionError
        If the filesystem denies write access.
    OSError
        If the parent directory cannot be created.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_text(Path("/tmp/output.txt"), "content")
    >>> read_text(Path("/tmp/output.txt"))
    'content'
    """
    try:
        ensure_dir(path.parent, exist_ok=True)
    except (PermissionError, OSError):
        raise
    try:
        path.write_text(data, encoding=encoding)
    except PermissionError:
        raise


def atomic_write(
    path: Path,
    data: str | bytes,
    mode: Literal["text", "binary"] = "text",
) -> None:
    """Write data atomically using a temporary file and rename.

    <!-- auto:docstring-builder v1 -->

    Parameters
    ----------
    path : Path
        Final file path to write.
    data : str | bytes
        Content to write (str for text mode, bytes for binary).
    mode : Literal['text', 'binary'], optional
        Write mode. Defaults to "text".
        Defaults to ``'text'``.

    Raises
    ------
    ValueError
        If mode is "text" but data is bytes, or mode is "binary" but data is str.
    PermissionError
        If the filesystem denies write access.

    Examples
    --------
    >>> from pathlib import Path
    >>> atomic_write(Path("/tmp/atomic.txt"), "safe content")
    >>> read_text(Path("/tmp/atomic.txt"))
    'safe content'
    """
    try:
        ensure_dir(path.parent, exist_ok=True)
    except PermissionError:
        raise
    tmp_path: Path | None = None
    try:
        dir_arg = str(path.parent) if path.parent else None
        if mode == "text":
            if not isinstance(data, str):
                msg = "text mode requires str data"
                raise ValueError(msg)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=dir_arg,
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                tmp_path = Path(temp_file.name)
                temp_file.write(data)
                temp_file.flush()
        else:
            if not isinstance(data, bytes):
                msg = "binary mode requires bytes data"
                raise ValueError(msg)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=dir_arg,
                delete=False,
            ) as temp_file:
                tmp_path = Path(temp_file.name)
                temp_file.write(data)
                temp_file.flush()
        if tmp_path is not None:
            try:
                tmp_path.replace(path)
            except PermissionError:
                raise
    finally:
        if tmp_path is not None and sys.exc_info()[0] is not None:
            tmp_path.unlink(missing_ok=True)
