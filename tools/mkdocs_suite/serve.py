"""Utility script to run MkDocs dev server on the first available port.

The default MkDocs behaviour is to bind to ``127.0.0.1:8000`` and error when the
address is already in use. This helper script probes a configurable port range
and automatically selects the first free port, delegating to ``mkdocs serve``
with the ``--dev-addr`` flag.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

from tools._shared.logging import get_logger
from tools._shared.proc import ToolExecutionError, run_tool

LOGGER = get_logger(__name__)

DEFAULT_CONFIG = Path("tools/mkdocs_suite/mkdocs.yml")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MIN_PORT = 8000
DEFAULT_MAX_PORT = 8019


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start mkdocs serve using the first available port in a range.",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to mkdocs configuration file.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Interface to bind the dev server to (default: %(default)s).",
    )
    parser.add_argument(
        "--min-port",
        type=int,
        default=DEFAULT_MIN_PORT,
        help="Lowest port to attempt (default: %(default)d).",
    )
    parser.add_argument(
        "--max-port",
        type=int,
        default=DEFAULT_MAX_PORT,
        help="Highest port to attempt (default: %(default)d).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Pass --open to mkdocs (open browser automatically).",
    )
    return parser.parse_args(argv)


def _find_available_port(host: str, min_port: int, max_port: int) -> int:
    """Return the first available TCP port for the given host.

    Parameters
    ----------
    host : str
        Hostname or IP address to bind.
    min_port : int
        Lowest port to probe (inclusive).
    max_port : int
        Highest port to probe (inclusive).

    Returns
    -------
    int
        The first available port number in the requested range.

    Raises
    ------
    ValueError
        If ``min_port`` exceeds ``max_port``.
    RuntimeError
        If host resolution fails or no ports are available in the range.
    """
    if min_port > max_port:
        msg = f"min_port {min_port} greater than max_port {max_port}"
        raise ValueError(msg)

    for port in range(min_port, max_port + 1):
        try:
            addrinfos = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:  # pragma: no cover - invalid host configuration
            msg = f"Failed to resolve host {host!r}"
            raise RuntimeError(msg) from exc

        for family, socktype, proto, _canonname, sockaddr in addrinfos:
            with socket.socket(family, socktype, proto) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(sockaddr)
                except OSError:
                    continue
            return port

    msg = f"No free ports found in range {min_port}-{max_port} for host {host}"
    raise RuntimeError(msg)


def find_available_port(host: str, min_port: int, max_port: int) -> int:
    """Return an available TCP port for ``host`` within ``min_port``-``max_port``.

    Parameters
    ----------
    host : str
        Hostname or IP address to bind.
    min_port : int
        Lowest port to probe (inclusive).
    max_port : int
        Highest port to probe (inclusive).

    Returns
    -------
    int
        The first available port number in the requested range.
    """
    return _find_available_port(host, min_port, max_port)


def main(argv: list[str] | None = None) -> int:
    """Run MkDocs dev server on the first available port.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code returned by the underlying ``mkdocs serve`` process.
    """
    args = _parse_args(argv or sys.argv[1:])

    port = find_available_port(args.host, args.min_port, args.max_port)
    dev_addr = f"{args.host}:{port}"
    LOGGER.info("Starting MkDocs dev server", extra={"dev_addr": dev_addr})

    command = [
        "mkdocs",
        "serve",
        "--config-file",
        str(args.config_file),
        "--dev-addr",
        dev_addr,
    ]
    if args.open:
        command.append("--open")

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    src_path = (repo_root / "src").resolve()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join((str(src_path), existing_pythonpath))
    else:
        env["PYTHONPATH"] = str(src_path)

    try:
        result = run_tool(command, env=env, check=False)
    except KeyboardInterrupt:  # pragma: no cover - interactive interrupt
        LOGGER.info("MkDocs dev server interrupted", extra={"dev_addr": dev_addr})
        return 130
    except ToolExecutionError as exc:
        LOGGER.exception(
            "MkDocs dev server failed",
            extra={
                "dev_addr": dev_addr,
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            },
        )
        return exc.returncode if exc.returncode is not None else 1

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
