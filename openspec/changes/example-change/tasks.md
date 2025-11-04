## 1. Implementation
- [ ] 1.1 Read `proposal.md` and `design.md`
- [ ] 1.2 Create/update JSON Schemas or OpenAPI if data crosses boundaries
- [ ] 1.3 Implement typed public API and pure logic first (no I/O coupling)
- [ ] 1.4 Integrate I/O and wiring (CLI/HTTP/jobs) after logic is tested

## 2. Testing
- [ ] 2.1 Unit tests (parametrized; happy paths + edge cases + negative cases)
- [ ] 2.2 Integration tests (if applicable; mark with `@pytest.mark.integration`)
- [ ] 2.3 Doctest/xdoctest examples execute successfully
- [ ] 2.4 Regression test added for any bug fixed

## 3. Docs & Artifacts
- [ ] 3.1 Update docstrings (NumPy style; coverage ≥ 90%)
- [ ] 3.2 `make artifacts` regenerates docs/nav/catalog without diff churn
- [ ] 3.3 Verify deep links (editor/GitHub) from Agent Portal

## 4. Rollout
- [ ] 4.1 Feature flags / config toggles (if any)
- [ ] 4.2 Migration steps (data, config, schema)
- [ ] 4.3 Observability (logs/metrics) and dashboards updated

## 5. Acceptance Gates (paste outputs into PR)
```bash
uv run ruff format && uv run ruff check --fix
uv run pyrefly check
uv run pyright --warnings --pythonversion=3.13
uv run pytest -q
make artifacts && git diff --exit-code
openspec validate example-change --strict
```

## 6. Sign‑off
- [ ] Domain owner review
- [ ] Implementation owner review
- [ ] CI green; artifacts uploaded (coverage, JUnit, docs/portal)
