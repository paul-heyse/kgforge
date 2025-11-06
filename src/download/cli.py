"""Download CLI adopted onto the shared tooling metadata contracts."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from tools import (
    CliEnvelope,
    CliEnvelopeBuilder,
    ProblemDetailsParams,
    build_problem_details,
    get_logger,
    render_cli_envelope,
    with_fields,
)

from download import cli_context
from kgfoundry_common.navmap_loader import load_nav_metadata

__all__ = [
    "app",
    "harvest",
]
__navmap__ = load_nav_metadata(__name__, tuple(__all__))


CLI_COMMAND = cli_context.CLI_COMMAND
CLI_OPERATION_IDS = cli_context.CLI_OPERATION_IDS
CLI_TITLE = cli_context.CLI_TITLE
CLI_INTERFACE_ID = cli_context.CLI_INTERFACE_ID

CLI_CONFIG = cli_context.get_cli_config()

REPO_ROOT = cli_context.REPO_ROOT
CLI_ENVELOPE_DIR = REPO_ROOT / "site" / "_build" / "cli"
CLI_ENVELOPE_PATH = CLI_ENVELOPE_DIR / "download.json"

HARVEST_OPERATION_ID = CLI_OPERATION_IDS["harvest"]
HARVEST_OVERRIDE = cli_context.get_operation_override("harvest")
if HARVEST_OVERRIDE and HARVEST_OVERRIDE.description:
    HARVEST_DESCRIPTION = HARVEST_OVERRIDE.description
else:
    HARVEST_DESCRIPTION = "Harvest documents from OpenAlex matching query parameters."
LOGGER = get_logger(__name__)

DEFAULT_YEARS = ">=2018"
DEFAULT_MAX_WORKS = 20_000


def _resolve_cli_help() -> str:
    title = CLI_CONFIG.title or CLI_TITLE
    version = CLI_CONFIG.version
    return f"{title} ({version})"


app = typer.Typer(help=_resolve_cli_help(), no_args_is_help=True, add_completion=False)
download_app = typer.Typer(help=HARVEST_DESCRIPTION, no_args_is_help=True, add_completion=False)
app.add_typer(download_app, name=CLI_COMMAND, help=HARVEST_DESCRIPTION)


def _write_cli_envelope(envelope: CliEnvelope) -> Path:
    CLI_ENVELOPE_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render_cli_envelope(envelope)
    CLI_ENVELOPE_PATH.write_text(rendered, encoding="utf-8")
    return CLI_ENVELOPE_PATH


def _harvest_problem(
    detail: str, *, status: int = 500, extras: dict[str, Any] | None = None
) -> dict[str, Any]:
    return build_problem_details(
        ProblemDetailsParams(
            type="https://kgfoundry.dev/problems/download/harvest-error",
            title="Download harvest command failed",
            status=status,
            detail=detail,
            instance=f"urn:cli:download:{CLI_INTERFACE_ID}",
            extensions=extras,
        )
    )


@download_app.command(help=HARVEST_DESCRIPTION)
def harvest(
    topic: Annotated[str, typer.Argument(help="Topic query string to harvest.")],
    years: Annotated[
        str,
        typer.Option(
            "--years",
            "-y",
            help="Year filter expression (e.g., '>=2018').",
            metavar="EXPR",
            show_default=True,
        ),
    ] = DEFAULT_YEARS,
    max_works: Annotated[
        int,
        typer.Option(
            "--max-works",
            "-m",
            help="Maximum number of works to harvest.",
            metavar="COUNT",
            show_default=True,
        ),
    ] = DEFAULT_MAX_WORKS,
) -> None:
    """Harvest documents from OpenAlex using the shared CLI tooling context.

    Raises
    ------
    typer.Exit
        Raised with a non-zero exit code when the command fails.
    """
    start = time.monotonic()
    builder = CliEnvelopeBuilder.create(command=CLI_COMMAND, status="success", subcommand="harvest")
    logger = with_fields(
        LOGGER,
        {
            "operation_id": HARVEST_OPERATION_ID,
            "topic": topic,
            "years": years,
            "max_works": max_works,
        },
    )

    logger.info("Harvest command started")
    try:
        message = f"[dry-run] would harvest topic={topic!r} years={years!r} max_works={max_works}"
        builder.add_file(path="openalex", status="success", message=message)
        typer.echo(message)
    except Exception as exc:  # pragma: no cover - defensive catch for future integrations
        problem = _harvest_problem(str(exc))
        builder.add_error(status="error", message=str(exc), problem=problem)
        builder.set_problem(problem)
        envelope = builder.finish(duration_seconds=time.monotonic() - start)
        path = _write_cli_envelope(envelope)
        logger.exception(
            "Harvest command failed",
            extra={"status": "error", "cli_envelope": str(path)},
        )
        raise typer.Exit(code=1) from exc

    envelope = builder.finish(duration_seconds=time.monotonic() - start)
    path = _write_cli_envelope(envelope)
    logger.info(
        "Harvest command completed",
        extra={
            "status": "success",
            "cli_envelope": str(path),
            "duration_seconds": envelope.duration_seconds,
        },
    )


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    app()
