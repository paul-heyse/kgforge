## 1. Planning & Analysis
- [ ] 1.1 Capture the current state: run `tools/update_docs.sh` and copy the ES01 warning
      block into a scratch file so we know exactly which symbols trigger problems today.
- [ ] 1.2 Review `tools/auto_docstrings.py` to understand how `extended_summary()` works and
      which helper functions (e.g., `_is_magic`, `_is_pydantic_field`) are already available.
- [ ] 1.3 Decide which Pydantic members we want to *document* with boilerplate text versus
      which ones we should suppress entirely (listing them out explicitly prevents surprises).

## 2. Implementation
- [ ] 2.1 Implement a stock-summary mapping inside `extended_summary()` for each group of
      magic methods we keep (comparisons, numeric, container protocol, lifecycle methods).
      Keep the text short, in imperative mood, and end with a period (NumPy style).
- [ ] 2.2 Update the Pydantic branch so that helpers such as `model_post_init`,
      `__pydantic_core_schema__`, etc. either return a boilerplate sentence (“Helper generated
      by Pydantic to expose the core schema.”) or are filtered out if we decide not to
      surface them.
- [ ] 2.3 Extend `process_file()` filters if we choose to hide any new members; otherwise
      ensure the summary text covers every helper surfaced in `tasks 1.3`.
- [ ] 2.4 Update or add snapshot/unit tests under `tests/` that call `build_docstring()` for a
      sample class containing magic methods and a Pydantic model, asserting the extended
      summary lines are present.

## 3. Validation
- [ ] 3.1 Run `make docstrings` locally; linting must pass without D4xx errors.
- [ ] 3.2 Run `tools/update_docs.sh` and verify the Sphinx output no longer contains ES01
      warnings for the magic/Pydantic symbols we covered (allowing us to target the next set
      of warnings separately).
- [ ] 3.3 If new warnings remain, document them in the change notes so the next iteration
      knows what is still outstanding.

