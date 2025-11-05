Awesome — here’s a **turn-key kit** that takes a Typer CLI and turns it into:

1. an **OpenAPI CLI spec** (`openapi/openapi-cli.yaml`) with rich `x-cli` metadata
2. a **MkDocs + ReDoc** page that renders it beautifully
3. a tiny **D2 diagram generator** that clusters operations by tag and deep-links into ReDoc
4. a **MkDocs Typer** page that renders your CLI help directly from code

I’ve written this so a junior dev can paste files in and run the commands.

---

# 0) What you’ll create

```
openapi/
  openapi-cli.yaml                # generated OpenAPI for your CLI
  _augment_cli.yaml               # optional tag/metadata map (data-only)
tools/
  typer_to_openapi_cli.py         # generator: Typer -> OpenAPI CLI
docs/
  api/openapi-cli.md              # ReDoc page (loads YAML above)
  cli/index.md                    # Typer docs page (renders CLI help)
  diagrams/index.md               # diagram landing page
  _scripts/gen_cli_diagram.py     # generator: tag-cluster D2 diagram
mkdocs.yml
```

---

# 1) Install the few libraries we’ll use

```bash
# Docs & rendering
pip install mkdocs mkdocs-material mkdocs-redoc-tag mkdocs-gen-files mkdocs-typer mkdocs-d2-plugin

# CLI stack
pip install typer click pyyaml

# Optional governance (highly recommended)
npm i -g @redocly/cli
```

---

# 2) MkDocs wiring (drop-in)

**mkdocs.yml**

```yaml
site_name: Your Project Docs
strict: true

theme:
  name: material

plugins:
  - search
  - redoc-tag
  - d2
  - gen-files:
      scripts:
        - docs/_scripts/gen_cli_diagram.py

markdown_extensions:
  - admonition
  - attr_list
  - pymdownx.details
  - pymdownx.superfences
  - mkdocs-typer

nav:
  - Overview: index.md
  - CLI:
      - cli/index.md
  - CLI Spec:
      - api/openapi-cli.md
  - Diagrams:
      - diagrams/index.md
```

**docs/api/openapi-cli.md**

```markdown
# CLI Spec (ReDoc)

<redoc src="./openapi-cli.yaml"/>
```

**docs/cli/index.md**

```markdown
# Command Reference

::: mkdocs-typer
    :module: your_pkg.cli     # change to your Typer module
    :command: app             # change to your Typer app variable
```

> If your Typer app is nested (e.g., `your_pkg.tools.cli:app`), update the two lines above and in the generator command below.

---

# 3) A tiny generator: **Typer → OpenAPI CLI**

Save as **tools/typer_to_openapi_cli.py**:

```python
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import sys
from typing import Any, Dict, List, Tuple
import yaml

# Typer uses Click under the hood
import click
try:
    from typer.main import get_command  # Typer -> click.Command
except Exception:
    get_command = None  # type: ignore[assignment]

HTTP_METHODS = {"get","post","put","delete","patch","head","options","trace"}

def import_object(path: str) -> Any:
    """
    Import "pkg.mod:attr" and return the attribute.
    If only "pkg.mod" is given, return the module.
    """
    if ":" in path:
        mod, attr = path.split(":", 1)
        module = importlib.import_module(mod)
        return getattr(module, attr)
    return importlib.import_module(path)

def to_click_command(obj: Any) -> click.core.Command:
    """Accept a Typer app or a click command/group and return click command."""
    if isinstance(obj, click.core.Command):
        return obj
    if get_command is None:
        raise RuntimeError("Typer not available; pass a click.Command or install typer.")
    return get_command(obj)

def snake_to_kebab(name: str) -> str:
    return name.replace("_", "-")

def param_schema(p: click.Parameter) -> Tuple[Dict[str, Any], bool, str]:
    """
    Return (schema, required, name_for_example) for a click Parameter.
    """
    # Base schema
    schema: Dict[str, Any] = {}
    required = getattr(p, "required", False)
    name_for_example = p.name

    # Type mapping
    t = getattr(p, "type", None)
    # default to string
    schema["type"] = "string"

    # choices
    if hasattr(t, "choices") and t.choices:
        schema["type"] = "string"
        schema["enum"] = list(t.choices)
    else:
        tname = getattr(t, "name", "").lower()
        if tname in {"int", "integer"}:
            schema["type"] = "integer"
        elif tname in {"float"}:
            schema["type"] = "number"
        elif tname in {"bool", "boolean"}:
            schema["type"] = "boolean"
        elif tname in {"path", "file", "filename"}:
            schema["type"] = "string"
            schema["format"] = "path"

    # multiple / nargs
    multiple = getattr(p, "multiple", False)
    nargs = getattr(p, "nargs", 1)
    if multiple or (isinstance(nargs, int) and nargs != 1) or nargs == -1:
        schema = {"type": "array", "items": schema}

    # description/help
    help_text = getattr(p, "help", None)
    if help_text:
        schema["description"] = help_text

    return schema, required, name_for_example

def build_example(bin_name: str, cmd_tokens: List[str], params: List[click.Parameter]) -> str:
    parts = [bin_name, *cmd_tokens]
    for p in params:
        # arguments vs options
        if p.param_type_name == "argument":
            parts.append(f"<{p.name}>")
        else:
            # option flags
            bool_flag = getattr(p, "is_flag", False)
            opt_name = f"--{snake_to_kebab(p.name)}"
            if bool_flag:
                parts.append(opt_name)
            else:
                parts.extend([opt_name, f"<{p.name}>"])
    return " ".join(parts)

def walk_commands(cmd: click.core.Command, tokens: List[str]) -> List[Tuple[List[str], click.core.Command]]:
    """
    Traverse click command tree. Returns list of (tokens, command) for leaf commands.
    Also include group callback as a runnable command if it has a callback.
    """
    out: List[Tuple[List[str], click.core.Command]] = []
    is_group = isinstance(cmd, click.core.Group)
    has_sub = is_group and getattr(cmd, "commands", {})
    if is_group and has_sub:
        # group callback itself (if any)
        if getattr(cmd, "callback", None):
            out.append((tokens, cmd))
        for name, sub in cmd.commands.items():  # type: ignore[attr-defined]
            out.extend(walk_commands(sub, tokens + [name]))
    else:
        out.append((tokens, cmd))
    return out

def make_openapi(
    click_cmd: click.core.Command,
    bin_name: str,
    title: str,
    version: str,
    augment: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build an OpenAPI 3.1 document (as dict) for the CLI.
    """
    augment = augment or {}
    tag_defs = {t["name"]: t for t in augment.get("tags", [])}
    tag_groups = augment.get("x-tagGroups")
    op_overrides: Dict[str, Any] = augment.get("operations", {})

    doc: Dict[str, Any] = {
        "openapi": "3.1.0",
        "info": {"title": title, "version": version},
        "tags": list(tag_defs.values()),
        "paths": {},
    }
    if tag_groups:
        doc["x-tagGroups"] = tag_groups

    ops = walk_commands(click_cmd, [])

    all_tags = set()

    for tokens, cmd in ops:
        if not tokens:
            # single-command app: give it a name
            tokens = [cmd.name or "run"]

        # path and operationId
        path = "/cli/" + "/".join(snake_to_kebab(t) for t in tokens)
        op_id = "cli." + ".".join(tokens)

        # choose a tag: by override, else first token
        override = op_overrides.get(op_id) or op_overrides.get(" ".join(tokens))
        tags = override.get("tags") if override else None
        if not tags:
            tags = [tokens[0]]
        for t in tags:
            all_tags.add(t)

        # parameters -> requestBody schema
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in getattr(cmd, "params", []):
            schema, is_required, _ = param_schema(p)
            props[p.name] = schema
            if is_required and not getattr(p, "is_flag", False) and getattr(p, "default", None) is None:
                required.append(p.name)

        req_schema: Dict[str, Any] = {"type": "object", "properties": props}
        if required:
            req_schema["required"] = sorted(set(required))

        # description & summary
        summary = getattr(cmd, "short_help", None) or (getattr(cmd, "help", "") or "").strip().split("\n")[0]
        description = (getattr(cmd, "help", "") or "").strip()

        # x-cli metadata
        example = build_example(bin_name, [snake_to_kebab(t) for t in tokens], list(getattr(cmd, "params", [])))
        x_cli = {
            "bin": bin_name,
            "command": " ".join(snake_to_kebab(t) for t in tokens),
            "examples": [example],
            "exitCodes": [{"code": 0, "meaning": "success"}],
        }
        if override:
            for k, v in override.items():
                if k.startswith("x-") or k in {"env"}:
                    x_cli[k] = v

        # assemble operation
        op = {
            "operationId": op_id,
            "tags": tags,
            "summary": summary or "Run CLI command",
            "description": description,
            "x-cli": x_cli,
            "requestBody": {
                "required": False if not props else True,
                "content": {"application/json": {"schema": req_schema}},
            },
            "responses": {
                "200": {
                    "description": "CLI result",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "stdout": {"type": "string"},
                                    "stderr": {"type": "string"},
                                    "exitCode": {"type": "integer"},
                                    "artifacts": {"type": "array", "items": {"type": "string"}},
                                },
                                "required": ["exitCode"],
                            }
                        }
                    },
                }
            },
        }

        doc["paths"].setdefault(path, {})
        doc["paths"][path]["post"] = op  # we model invocations as POST

    # ensure tags are present
    existing = {t["name"] for t in doc.get("tags", [])}
    for t in sorted(all_tags):
        if t not in existing:
            doc["tags"].append({"name": t, "description": f"Commands for {t}."})

    return doc

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate OpenAPI for a Typer/Click CLI.")
    ap.add_argument("--app", required=True, help="Import path to Typer or click app, e.g. your_pkg.cli:app")
    ap.add_argument("--bin", default="kgf", help="Binary name shown in examples (default: kgf)")
    ap.add_argument("--title", default="Project CLI", help="OpenAPI info.title")
    ap.add_argument("--version", default="0.1.0", help="OpenAPI info.version")
    ap.add_argument("--augment", default="openapi/_augment_cli.yaml", help="YAML with tags/x-tagGroups/operation overrides")
    ap.add_argument("--out", default="openapi/openapi-cli.yaml", help="Output OpenAPI YAML path")
    args = ap.parse_args()

    obj = import_object(args.app)
    cmd = to_click_command(obj)
    try:
        with open(args.augment, "r") as f:
            augment = yaml.safe_load(f) or {}
    except FileNotFoundError:
        augment = {}

    spec = make_openapi(cmd, args.bin, args.title, args.version, augment)
    # create parent dirs if missing
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    sys.exit(main())
```

Make it executable:

```bash
chmod +x tools/typer_to_openapi_cli.py
```

---

# 4) Optional: human-friendly tags & groups (data-only)

**openapi/_augment_cli.yaml** (edit freely; no code changes needed)

```yaml
tags:
  - name: ingest
    x-displayName: Ingest
    description: Commands that download, parse, and embed data.
  - name: search
    x-displayName: Search
    description: Local/hybrid search and evaluation.
  - name: admin
    x-displayName: Admin
    description: Maintenance and housekeeping tasks.

x-tagGroups:
  - name: Core
    tags: [ingest, search]
  - name: Operations
    tags: [admin]

# You can override by operationId ("cli.ingest.files") or by command string ("ingest files")
operations:
  cli.ingest.files:
    tags: [ingest]
    x-env: [KGF_PROFILE]
    x-handler: "your_pkg.cli.ingest:files"
    x-codeSamples:
      - lang: bash
        source: "kgf ingest files --src ./docs --profile default"
  "ingest urls":
    tags: [ingest]
    x-handler: "your_pkg.cli.ingest:urls"
  "search run":
    tags: [search]
    x-handler: "your_pkg.cli.search:run"
```

> You can add `x-codeSamples`, `x-handler`, and `x-env` here. The generator merges them into each operation’s `x-cli` block.

---

# 5) Generate the spec from your Typer app

```bash
# Example: your Typer app lives at your_pkg/cli.py with variable `app`
./tools/typer_to_openapi_cli.py \
  --app your_pkg.cli:app \
  --bin kgf \
  --title "KGFoundry CLI" \
  --version "0.1.0"
```

You’ll get **openapi/openapi-cli.yaml**. (If you use Redocly, you can `redocly lint`/`bundle` it too, but the ReDoc page will happily load it as-is.)

---

# 6) Tiny D2 generator: tag-cluster diagram with deep links

**docs/_scripts/gen_cli_diagram.py**

```python
from __future__ import annotations
import yaml
import mkdocs_gen_files

SPEC_PATH = "openapi/openapi-cli.yaml"  # same file your ReDoc page loads

spec = yaml.safe_load(open(SPEC_PATH))
ops = []  # (tag, method, path, operationId, summary)

for path, item in (spec.get("paths") or {}).items():
    for method, op in (item or {}).items():
        if method.lower() != "post":
            continue
        opid = op.get("operationId")
        tags = op.get("tags") or ["cli"]
        summary = op.get("summary", "")
        for tag in tags:
            ops.append((tag, method.upper(), path, opid, summary))

# Write a single diagram grouped by tag
with mkdocs_gen_files.open("diagrams/cli_by_tag.d2", "w") as d:
    d.write('direction: right\nCLI: "CLI" {\n')
    for tag in sorted({t for (t, *_rest) in ops}):
        d.write(f'  "{tag}": "{tag}" {{}}\n')
    for tag, method, path, opid, summary in ops:
        node = f'{method} {path}'
        label = f'{method} {path}\\n{summary}'.strip()
        d.write(f'  "{node}": "{label}" {{ link: "../api/openapi-cli/#operation/{opid}" }}\n')
        d.write(f'  "{tag}" -> "{node}"\n')
    d.write('}\n')

# Landing page
with mkdocs_gen_files.open("diagrams/index.md", "w") as f:
    f.write("# Diagrams\n\n- [CLI by Tag](./cli_by_tag.d2)\n")
```

This runs during `mkdocs build/serve` (via `gen-files`) and creates `diagrams/cli_by_tag.d2`. Nodes link directly into your ReDoc section for each operation (`#operation/<operationId>`).

---

# 7) (Optional) A minimal, concrete example output

If your Typer app looks like:

```python
# your_pkg/cli.py
import typer
app = typer.Typer(help="KGFoundry CLI")

@app.command(help="Ingest files from a directory")
def files(src: str = typer.Option(..., help="Path to source"),
          profile: str = typer.Option("default", help="Ingest profile")):
    ...

@app.command(help="Run a local search over the index")
def search(query: str = typer.Argument(..., help="Query string"),
           k: int = typer.Option(10, help="Top-k results")):
    ...
```

The generator will emit operations like:

```yaml
openapi: 3.1.0
info: { title: KGFoundry CLI, version: 0.1.0 }
tags:
  - name: ingest
    x-displayName: Ingest
    description: Commands that download, parse, and embed data.
  - name: search
    x-displayName: Search
    description: Local/hybrid search and evaluation.
x-tagGroups:
  - name: Core
    tags: [ingest, search]
paths:
  /cli/files:
    post:
      operationId: cli.files
      tags: [ingest]
      summary: Ingest files from a directory
      x-cli:
        bin: kgf
        command: files
        examples: ["kgf files --src <src> --profile <profile>"]
        exitCodes: [{code: 0, meaning: success}]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                src: { type: string, format: path, description: "Path to source" }
                profile: { type: string, description: "Ingest profile" }
              required: [src]
      responses: { "200": { description: "CLI result", content: { application/json: { schema: { type: object, properties: { stdout: {type: string}, stderr: {type: string}, exitCode: {type: integer}, artifacts: {type: array, items: {type: string}} }, required: [exitCode] }}}}}
  /cli/search:
    post:
      operationId: cli.search
      tags: [search]
      summary: Run a local search over the index
      x-cli:
        bin: kgf
        command: search
        examples: ["kgf search <query> --k <k>"]
        exitCodes: [{code: 0, meaning: success}]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query: { type: string, description: "Query string" }
                k: { type: integer, description: "Top-k results" }
              required: [query]
      responses: { "200": { description: "CLI result", content: { application/json: { schema: { type: object, properties: { stdout: {type: string}, stderr: {type: string}, exitCode: {type: integer}, artifacts: {type: array, items: {type: string}} }, required: [exitCode] }}}}}
```

---

# 8) Build & view

```bash
# 1) Generate the OpenAPI file from your Typer app
./tools/typer_to_openapi_cli.py --app your_pkg.cli:app --bin kgf --title "KGFoundry CLI" --version "0.1.0"

# (optional but recommended)
redocly lint openapi/openapi-cli.yaml

# 2) Serve docs
mkdocs serve
```

* Navigate to **CLI → Command Reference** for live Typer docs from code.
* Navigate to **CLI Spec** to see the ReDoc reference.
* Click **Diagrams → CLI by Tag** for the graph; nodes deep-link into ReDoc.

---

## Notes & tweaks

* **Tags & groups**: put the human-friendly names and descriptions in `_augment_cli.yaml` so you don’t have to repeat them in code.
* **x-cli**: add `x-handler`, `x-env`, `x-codeSamples` per operation (either in `_augment_cli.yaml` or by extending the generator to look up your own metadata).
* **Operation IDs**: the generator uses `cli.<cmd...>`; feel free to add a rule in CI (Redocly) to ensure uniqueness and URL safety.
* **Return schema**: I modeled a simple `{stdout, stderr, exitCode, artifacts}` shape; if your CLI returns JSON payloads, swap in your schema.

If you want me to tailor the generator defaults (tags, groups, `x-cli` fields) to your real command tree (e.g., `kgfoundry.cli:app` with `ingest`, `registry`, `search`, etc.), just tell me the import path to your Typer app and the preferred binary name—I’ll produce a repo-ready version with your exact tags and a starter `_augment_cli.yaml`.
