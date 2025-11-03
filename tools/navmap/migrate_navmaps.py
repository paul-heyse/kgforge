"""Regenerate the navigation map JSON consumed by the documentation site."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from tools import get_logger
from tools.navmap.build_navmap import build_index

LOGGER = get_logger(__name__)

DEFAULT_OUTPUT: Final[Path] = (
    Path(__file__).resolve().parents[2] / "site" / "_build" / "navmap.json"
)


@dataclass(frozen=True)
class NavmapWriteConfig:
    """Configuration controlling how the navmap JSON payload is emitted."""

    indent: int | None = 2


@dataclass(frozen=True)
class MigrateArgs:
    """CLI arguments for ``migrate_navmaps`` after parsing."""

    output: Path
    write_config: NavmapWriteConfig


def migrate_navmaps(
    output: Path | None = None,
    *,
    config: NavmapWriteConfig | None = None,
) -> dict[str, object]:
    """Rebuild the navigation map JSON file from the current source tree.

    Parameters
    ----------
    output
        Destination path for the generated JSON document. When ``None`` the
        navmap is only returned to the caller.
    config
        Controls JSON emission options, including indentation. Defaults to
        :class:`NavmapWriteConfig` which emits human-readable JSON.

    Returns
    -------
    dict[str, object]
        Structured navigation metadata emitted by
        :func:`tools.navmap.build_navmap.build_index`.
    """
    write_config = config or NavmapWriteConfig()
    index = build_index()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(index, indent=write_config.indent)
        output.write_text(text, encoding="utf-8")
    return index


def _parse_args(argv: list[str] | None = None) -> MigrateArgs:
    """Parse CLI arguments for the navmap migration utility."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated navmap JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit JSON without indentation to save space.",
    )
    namespace = parser.parse_args(argv)
    return MigrateArgs(
        output=cast("Path", namespace.output),
        write_config=NavmapWriteConfig(indent=None if cast("bool", namespace.compact) else 2),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for regenerating the navmap JSON asset.

    Parameters
    ----------
    argv
        Optional argument vector, primarily used by tests.

    Returns
    -------
    int
        ``0`` on success so the helper integrates cleanly with shell pipelines.
    """
    args = _parse_args(argv)
    migrate_navmaps(args.output, config=args.write_config)
    LOGGER.info("Wrote navmap index to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
