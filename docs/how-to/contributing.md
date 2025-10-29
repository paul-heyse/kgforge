# Contributing Docstrings and Documentation

The documentation pipeline now enforces strict **NumPy-style docstrings** across the
entire project. Follow the guidelines below whenever you add or update code so that
pre-commit hooks and the automated documentation build succeed.

## Required Sections

Every public function **must** include the following sections (omit only when empty):

- `Parameters` with entries of the form ``name : type`` and `, optional` when defaults exist
- `Returns` describing the return type/value (skip if the function returns `None`)
- `Raises` listing each exception that can be raised and the condition
- `Examples` containing at least one doctestable block using `>>>`
- `See Also` referencing related functions, classes, or modules
- `Notes` summarising constraints, complexity, or behavioural caveats

Public classes **must** also document an `Attributes` section and list each public
method under `Methods`. When no attributes exist, include `None` with a short explanation.

Module docstrings should start with an imperative summary and may include `Notes`,
`See Also`, or `References`. Custom sections such as `NavMap:` are no longer allowed.

## Templates

Docstring generation now runs through `python -m tools.docstring_builder update`.
The builder emits NumPy-style sections automatically, but when writing manual
docstrings you should mirror the structure below so re-runs remain idempotent:

```python
"""Summarise the function in the imperative mood.

Parameters
----------
arg : type
    Concise description of the argument.

Returns
-------
return_type
    Explain what is returned.

Raises
------
ExceptionType
    State the condition that triggers the exception.

Examples
--------
>>> from package.module import function
>>> result = function(...)
>>> result  # doctest: +ELLIPSIS
...

See Also
--------
package.module.other_function

Notes
-----
Mention constraints, complexity, or other important behaviour.
"""
```

## Validation Pipeline

The following tools run automatically in CI and via `pre-commit`:

- `pydoclint --style numpy src` enforces parameter/return parity
- `pydocstyle` checks NumPy docstring formatting
- `interrogate -i src --fail-under 90` maintains 90% coverage
- `python tools/update_navmaps.py` aborts if any docstring still contains `NavMap:`

Run these locally before pushing:

```bash
make docstrings            # Regenerate docstrings using the latest templates
pre-commit run --all-files # Run the full docstring validation suite
pytest --doctest-modules src
```

Refer to the [NumPy docstring style guide](https://numpydoc.readthedocs.io/en/latest/format.html)
for additional examples and best practices.
