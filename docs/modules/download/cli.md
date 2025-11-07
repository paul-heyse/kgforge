# download.cli

Downloader command suite that sources external corpora (currently OpenAlex) using the shared
CLI tooling contracts. Emits structured envelopes and metadata so downstream tooling (OpenAPI,
diagrams, documentation) remains in sync without bespoke glue.

[View source on GitHub](https://github.com/paul-heyse/kgfoundry/blob/main/src/download/cli.py)

## Hierarchy

- **Parent:** [download](../download.md)

## Sections

- **Public API**

## Contents

### download.cli._emit_envelope

::: download.cli._emit_envelope

### download.cli._envelope_path

::: download.cli._envelope_path

### download.cli._harvest_problem

::: download.cli._harvest_problem

### download.cli._resolve_cli_help

::: download.cli._resolve_cli_help

### download.cli.harvest

::: download.cli.harvest

## Related API operations

[`cli.download.harvest`](../../api/openapi-cli.md#operation/cli.download.harvest) (download-cli · CLI Spec)

## Relationships

**Imports:** `__future__.annotations`, [download.cli_context](cli_context.md), `kgfoundry_common.logging.LoggerAdapter`, `kgfoundry_common.navmap_loader.load_nav_metadata`, `logging`, `pathlib.Path`, `time`, `tools.CliEnvelope`, `tools.CliEnvelopeBuilder`, `tools.ProblemDetailsDict`, `tools.ProblemDetailsParams`, `tools.build_problem_details`, `tools.get_logger`, `tools.render_cli_envelope`, `tools.with_fields`, `typer`, `typing.Any`, `uuid.uuid4`

## Autorefs Examples

- [download.cli._emit_envelope][]
- [download.cli._envelope_path][]
- [download.cli._harvest_problem][]

## Neighborhood

```d2
direction: right
"download.cli": "download.cli" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/cli.py" }
"__future__.annotations": "__future__.annotations"
"download.cli" -> "__future__.annotations"
"download.cli_context": "download.cli_context" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/cli_context.py" }
"download.cli" -> "download.cli_context"
"kgfoundry_common.logging.LoggerAdapter": "kgfoundry_common.logging.LoggerAdapter"
"download.cli" -> "kgfoundry_common.logging.LoggerAdapter"
"kgfoundry_common.navmap_loader.load_nav_metadata": "kgfoundry_common.navmap_loader.load_nav_metadata"
"download.cli" -> "kgfoundry_common.navmap_loader.load_nav_metadata"
"logging": "logging"
"download.cli" -> "logging"
"pathlib.Path": "pathlib.Path"
"download.cli" -> "pathlib.Path"
"time": "time"
"download.cli" -> "time"
"tools.CliEnvelope": "tools.CliEnvelope"
"download.cli" -> "tools.CliEnvelope"
"tools.CliEnvelopeBuilder": "tools.CliEnvelopeBuilder"
"download.cli" -> "tools.CliEnvelopeBuilder"
"tools.ProblemDetailsDict": "tools.ProblemDetailsDict"
"download.cli" -> "tools.ProblemDetailsDict"
"tools.ProblemDetailsParams": "tools.ProblemDetailsParams"
"download.cli" -> "tools.ProblemDetailsParams"
"tools.build_problem_details": "tools.build_problem_details"
"download.cli" -> "tools.build_problem_details"
"tools.get_logger": "tools.get_logger"
"download.cli" -> "tools.get_logger"
"tools.render_cli_envelope": "tools.render_cli_envelope"
"download.cli" -> "tools.render_cli_envelope"
"tools.with_fields": "tools.with_fields"
"download.cli" -> "tools.with_fields"
"typer": "typer"
"download.cli" -> "typer"
"typing.Any": "typing.Any"
"download.cli" -> "typing.Any"
"uuid.uuid4": "uuid.uuid4"
"download.cli" -> "uuid.uuid4"
"download": "download" { link: "https://github.com/paul-heyse/kgfoundry/blob/main/src/download/__init__.py" }
"download" -> "download.cli" { style: dashed }
```

