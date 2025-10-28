# Contributing Sphinx-Gallery Examples

Sphinx-Gallery builds our example gallery directly from the Python modules in
`examples/`. Every file must follow a predictable docstring structure so the
generated pages have stable titles and cross-reference anchors. This guide walks
through the required format, explains the validation tooling, and provides
troubleshooting tips for common issues.

## Quick start checklist

1. Copy the docstring template below.
2. Update the title and description to match your example.
3. Tag the example with relevant keywords using the ``.. tags::`` directive.
4. Describe resource expectations inside the ``Constraints`` section.
5. Run `python tools/validate_gallery.py --verbose` before committing.
6. Execute `pytest --doctest-modules examples/<your-file>.py` to confirm code
   snippets still run.

## Docstring template {#docstring-template}

Each example begins with a module-level docstring that Sphinx-Gallery parses to
create the page heading. The first non-empty line becomes the HTML title and is
also used to derive the `sphx_glr_` reference label.

```python
"""My concise title (≤79 characters)
====================================

One or two sentences describing what the example demonstrates.

.. tags:: topic-one, topic-two

Constraints
-----------

- Time: <2s
- GPU: no
- Network: no

>>> # Optional doctest snippets
>>> answer = 40 + 2
>>> answer
42
"""
```

### Title and underline

- The first line must be plain text with no trailing period.
- The underline must consist only of `=` characters and match the title length
  within ±1 character.
- Add a blank line between the underline and the body text.

### Tags directive

Use ``.. tags::`` to help readers (and search) understand the categories an
example belongs to. Separate multiple tags with commas. The validator treats the
directive as mandatory.

### Constraints section

Summarise runtime expectations in a dedicated `Constraints` section using a
dashed underline. List each constraint as a bullet item such as execution time,
GPU usage, or network access.

### Doctests

Include `>>>` prompts when demonstrating Python usage. All examples are
doctested via `pytest --doctest-modules examples/`, so keep snippets small and
deterministic.

## Validation tooling

The `tools/validate_gallery.py` script enforces the required structure. It is
installed as a pre-commit hook and runs automatically whenever gallery files are
staged, but you can also invoke it manually:

```bash
python tools/validate_gallery.py --verbose
```

### CLI options

- `--examples-dir PATH` – Validate a different directory (default: `examples/`).
- `--strict` – Enable additional checks, such as rejecting titles that end with
  a period.
- `--verbose` – Print confirmation for passing files.
- `--fix` – Reserved for future automatic formatting helpers.

The script exits with code `0` when all examples pass or `1` when it reports any
violations. Missing directories or unsupported flags cause exit code `2`.

### Common validation errors

| Error message | How to fix |
| --- | --- |
| `docstring is empty` | Ensure the file starts with a module docstring. |
| `title exceeds 79 characters` | Shorten the first line of the docstring. |
| `title underline must be composed of '=' characters` | Replace decorative underlines with `=` characters. |
| `remove ':orphan:' directive` | Delete legacy `:orphan:` directives—Sphinx-Gallery adds anchors automatically. |
| `add a '.. tags::' directive` | Insert a tags directive underneath the description. |
| `add a 'Constraints' section with a dashed underline` | Document runtime requirements under a `Constraints` heading. |

## Local testing workflow

1. Run `python tools/validate_gallery.py --verbose` after editing examples.
2. Execute `pytest --doctest-modules examples/<file>.py` to run doctests for a
   single example.
3. Execute `pytest --doctest-modules examples/` to run the entire suite.
4. Build the docs with `tools/update_docs.sh` to confirm Sphinx renders pages
   without warnings.

## Debugging Sphinx-Gallery issues

- Use `sphinx-build -b html docs docs/_build/html -n` to surface unresolved
  references.
- Inspect the generated reStructuredText in `docs/gallery/` when titles or
  anchors look incorrect.
- Ensure every gallery file imports quickly and avoids heavyweight dependencies.
- Keep example output deterministic so doctest snapshots remain stable.

For a full overview of the automated documentation pipeline, see the
repository's `README-AUTOMATED-DOCUMENTATION.md`.
