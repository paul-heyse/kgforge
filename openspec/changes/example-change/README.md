# Example Change Template

This folder shows the expected structure for a new OpenSpec change.

```
openspec/changes/example-change/
├── proposal.md
├── tasks.md
├── design.md
└── specs/
    └── example-capability/
        └── spec.md
```

**How to use**

1. `cp -r openspec/changes/example-change openspec/changes/<your-change-id>`
2. Fill `proposal.md` (Why / What / Impact / Acceptance)
3. Replace the `example-capability` folder with your capability name(s)
4. Author deltas in `specs/<capability>/spec.md` using ADDED / MODIFIED / REMOVED / RENAMED blocks
5. Run:
   ```bash
   openspec validate <your-change-id> --strict
   uv run ruff format && uv run ruff check --fix
   uv run pyrefly check && uv run mypy --config-file mypy.ini
   uv run pytest -q
   make artifacts && git diff --exit-code
   ```
6. Open a PR from `openspec/<your-change-id>` and link these files in the description
