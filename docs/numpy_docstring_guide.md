# NumPy (numpydoc) docstrings: an exhaustive, best-in-class guide

Below is a practical, “do-it-right-the-first-time” guide to writing NumPy-style docstrings that render beautifully in Sphinx and are friendly to IDEs, linters, and static type checkers.

---

## Core principles (what “good” looks like)

* **Use the numpydoc standard** (the convention followed by NumPy/SciPy). Headings are underlined with hyphens, sections appear in a fixed order, and lines ideally wrap at ~75 characters for terminal readability. ([numpydoc.readthedocs.io][1])
* **Prefer human readability** over Sphinx quirks; numpydoc is designed to keep docstrings readable in plain text and inside `help()`. ([numpydoc.readthedocs.io][1])
* **Use Sphinx + the `numpydoc` extension** when building docs so NumPy-style sections parse correctly (don’t rely on plain Sphinx alone). ([numpy.org][2])
* **Even with type annotations, include types in the docstring sections** (NumPy style expects them; Sphinx’s example explicitly duplicates the return type). ([sphinx-doc.org][3])
* **When documenting methods, never list `self`**; keep the detailed docs on the function if a function/method pair exist and reference it from the method. ([numpydoc.readthedocs.io][1])

---

## The canonical section order (functions)

Each heading is underlined with `-`, without a blank line after the heading. Keep this order: ([numpydoc.readthedocs.io][1])

1. **Short Summary** – one sentence, no parameter names.
2. **Deprecation warning** – use the `.. deprecated::` directive and explain version/reason/replacement. ([numpydoc.readthedocs.io][1])
3. **Extended Summary** – a few sentences for context (no parameter details). ([numpydoc.readthedocs.io][1])
4. **Parameters** – name, type, and description (see details below). ([numpydoc.readthedocs.io][1])
5. **Returns** – types (name optional) or **Yields** for generators; **Receives** if you use `.send()`. ([numpydoc.readthedocs.io][1])
6. **Other Parameters** – rarely used kwargs to keep **Parameters** clean. ([numpydoc.readthedocs.io][4])
7. **Raises** / **Warns** / **Warnings** – exceptions emitted, warnings emitted, free-text cautions. ([numpydoc.readthedocs.io][1])
8. **See Also** – related callables with optional short descriptions; prefer concise links. ([numpydoc.readthedocs.io][1])
9. **Notes** – background, math (LaTeX ok but use sparingly); **References** – numbered bibliography used by *Notes*. ([numpydoc.readthedocs.io][1])
10. **Examples** – doctest-style snippets; separate examples with blank lines; mark nondeterministic output with `#random`. ([numpydoc.readthedocs.io][1])

> Tip: Section headers come immediately followed by content (no blank line), which is consistent with Numpydoc expectations and saves vertical space. ([developer.lsst.io][5])

---

## Parameters, types, shapes, and defaults (the tight way)

* **Syntax basics.** Use `name : type`; a space before the colon; if you omit the type, omit the colon. Surround parameter names in backticks when referenced in prose. ([numpydoc.readthedocs.io][1])
* **`optional` vs `default=`.** Either “`x : int, optional`” (then describe default in text) or “`x : int, default 10`”. If the default wouldn’t be a value users pass (e.g., `None` as sentinel), prefer `optional`. ([numpydoc.readthedocs.io][1])
* **Enums.** Fixed choices in braces: `order : {'C', 'F', 'A'}`; put the default first. ([numpydoc.readthedocs.io][1])
* **Combine identical parameters.** `x1, x2 : array_like` if type/shape/semantics match. ([numpydoc.readthedocs.io][1])
* **Varargs & kwargs.** Document as `*args` / `**kwargs` without a type on the colon line; explain in the text. ([numpydoc.readthedocs.io][1])

### Shapes & array-ish inputs

Numpydoc doesn’t mandate a single shape notation, but a widely adopted convention (from scikit-learn) is:

```
X : array-like of shape (n_samples, n_features)
y : array-like of shape (n_samples,)
```

Use parentheses for shapes, name dimensions clearly, and prefer “array-like” when you accept lists/tuples/NumPy arrays/sparse matrices. ([scikit-learn.org][6])

### DTypes and modern typing aliases

When you accept NumPy arrays or array-likes, pair your docstring types with **PEP 484** annotations using `numpy.typing`:

* `npt.ArrayLike`, `npt.NDArray`, `npt.DTypeLike` are the go-to aliases. ([numpy.org][7])

> Example (annotation + docstring):
>
> ```py
> import numpy as np
> import numpy.typing as npt
>
> def row_norms(X: npt.ArrayLike, *, dtype: npt.DTypeLike = np.float64) -> npt.NDArray[np.float64]:
>     """Row-wise ℓ2 norms.
>
>     Parameters
>     ----------
>     X : array-like of shape (n_samples, n_features)
>         Input data convertible to ``ndarray``.
>     dtype : data-type, default float64
>         Accumulator dtype.
>
>     Returns
>     -------
>     ndarray of shape (n_samples,)
>         ℓ2 norm of each row.
>     """
> ```
>
> NumPy style still expects you to include the type in the docstring even if you annotate the function signature. ([sphinx-doc.org][3])

---

## Returns / Yields / Receives

* **Returns.** Type is required; a name is optional. You may list multiple outputs as “name : type” entries. ([numpydoc.readthedocs.io][1])
* **Yields.** Same rules as **Returns** but for generators. **Receives** documents `.send()` input if you use coroutines. If you include **Receives**, you must also include **Yields**. ([numpydoc.readthedocs.io][1])

---

## Raises, Warns, Warnings

* **Raises** — list exceptions and when they occur (use judiciously).
* **Warns** — list warnings your function emits and when.
* **Warnings** — free-text cautions to users (not an emitted warning). ([numpydoc.readthedocs.io][1])

---

## See Also (done right)

Reference related callables; omit module prefixes for same submodule, include them for others; descriptions are optional and should be brief. You can wrap the description onto a new line if too long. ([numpydoc.readthedocs.io][1])

---

## Notes, math, references

Use **Notes** for background, algorithms, and optional LaTeX math (prefer code or pseudocode when possible; keep math readable). **References** are numbered footnotes supporting material cited in **Notes**; avoid ephemeral web links. ([numpydoc.readthedocs.io][1])

---

## Examples (doctest-friendly)

* Use `>>>` prompts and blank lines between examples; separate commentary with blank lines too.
* For nondeterministic output (e.g., RNG), mark with `#random`. ([numpydoc.readthedocs.io][1])

---

## Classes, modules, methods, and constants

* **Class docstrings**: include constructor **Parameters** there; optionally add **Attributes** for non-method attributes. ([numpydoc.readthedocs.io][1])
* **Method docstrings**: don’t list `self`; keep them brief and point to the function doc if there’s a function/method pair. ([numpydoc.readthedocs.io][1])
* **Modules**: summary plus optional sections (routine listings, see also, notes, references, examples). ([numpydoc.readthedocs.io][1])
* **Constants**: minimal sections as relevant; they’ll show up in Sphinx even if not in terminal help. ([numpydoc.readthedocs.io][1])

---

## Deprecations and versioning markers

* Prefer Sphinx directives in your docstrings:

  * `.. deprecated:: X.Y` with reason and replacement.
  * `.. versionadded:: X.Y` / `.. versionchanged:: X.Y` to mark lifecycle changes. ([numpydoc.readthedocs.io][1])

---

## Whitespace, punctuation, and markup “gotchas”

* Keep lines ~75 chars; wrap prose naturally. ([numpydoc.readthedocs.io][1])
* Enclose **parameter names** in single backticks; use **double backticks** for code literals (monospace). ([numpydoc.readthedocs.io][1])
* Triple-quoted docstrings use double quotes per PEP 257 / PEP 8 convention. ([Python Enhancement Proposals (PEPs)][8])

---

## Sphinx toolchain choices

* **Use `numpydoc`** for the strict NumPy spec; Sphinx’s **Napoleon** also supports NumPy style, but it has behavioral differences and is not the spec NumPy uses. If you want maximum fidelity to NumPy’s rules, prefer `numpydoc`. ([numpy.org][2])
* Core Sphinx extensions you’ll likely want: `sphinx.ext.autodoc`, `sphinx.ext.autosummary`, `sphinx.ext.napoleon` *(if you choose it instead of numpydoc)*, `sphinx.ext.intersphinx`, `sphinx.ext.doctest`, `sphinx.ext.coverage`. ([sphinx-doc.org][9])

---

## Linting & validation (make quality measurable)

* **numpydoc validator**: `python -m numpydoc --validate your.module.Object` or `numpydoc validate` to catch section order, capitalization, and other style errors; use as a pre-commit hook. ([numpydoc.readthedocs.io][10])
* **pydocstyle**: enable the **numpy convention** to check numpydoc-style basics across your codebase. ([pydocstyle.org][11])
* **Ruff docstring rules** (pydoclint in Ruff): enable Ruff’s preview features to get modern docstring checks (missing/extra exceptions, section ordering, etc.). ([docs.astral.sh][12])
* **numpydoc-linter**: AST-based file linter that emits numpydoc validate codes across files. ([PyPI][13])

**Example `pyproject.toml` snippets**

```toml
[tool.ruff]
target-version = "py313"

[tool.ruff.lint]
# Enable docstring checks that are still in preview in Ruff.
preview = true

# Example: enable pydocstyle (D) and pydoclint-derived (DOC) rules, plus others you use.
select = ["E", "F", "D", "DOC"]
ignore = [
  # Add project-specific ignores here
]

[tool.pydocstyle]
convention = "numpy"   # enforce numpydoc-style basic rules
```

**Example pre-commit hook**

```yaml
repos:
  - repo: https://github.com/pycqa/pydocstyle
    rev: 6.3.0
    hooks:
      - id: pydocstyle
        args: [--convention=numpy]

  - repo: https://github.com/numpy/numpydoc
    rev: v1.6.0
    hooks:
      - id: numpydoc-validation
```

(Use current tags as appropriate; the important bit is running validate in CI and locally.) ([pydocstyle.org][14])

---

## A “gold standard” function example

```py
import numpy as np
import numpy.typing as npt

def standardize(
    X: npt.ArrayLike,
    *,
    axis: int | tuple[int, ...] = 0,
    ddof: int = 0,
) -> npt.NDArray[np.floating]:
    """Standardize an array along a given axis.

    Subtracts the mean and divides by the standard deviation along ``axis``.
    NaNs are propagated.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) or (n_samples,)
        Input data convertible to ``ndarray``.
    axis : int or tuple of int, default 0
        Axis (or axes) along which to compute statistics.
    ddof : int, default 0
        Delta degrees of freedom for the standard deviation.

    Returns
    -------
    ndarray
        Standardized values with the same shape as ``X``.

    Raises
    ------
    ZeroDivisionError
        If the standard deviation is zero along the requested axis.

    See Also
    --------
    np.mean : Arithmetic mean.
    np.std : Standard deviation.

    Notes
    -----
    Let :math:`x` be the data and :math:`\mu, \sigma` the sample mean and
    standard deviation along ``axis``. The output is
    :math:`(x - \mu)/\sigma`.

    Examples
    --------
    >>> standardize([1., 2., 3.])
    array([-1.22474487, 0., 1.22474487])
    """
```

This specimen applies the ordering, headings, shapes, `ArrayLike`/`NDArray` typing, backticks for names, and a proper **Raises / See Also / Notes / Examples** set.

---

## A “gold standard” class example (constructor + attributes)

```py
class OnlineScaler:
    """Online standardization via Welford's algorithm.

    Parameters
    ----------
    ddof : int, default 0
        Delta degrees of freedom.
    clip : float or None, optional
        If given, clip standardized values to ``[-clip, clip]``.

    Attributes
    ----------
    n_ : int
        Number of samples seen.
    mean_ : float
        Running mean.
    var_ : float
        Running variance.

    See Also
    --------
    standardize : Batch standardization.

    Examples
    --------
    >>> s = OnlineScaler()
    >>> s.update(1.0); s.update(2.0); s.update(3.0)
    >>> round(s.transform(2.0), 6)
    0.0
    """
```

This mirrors numpydoc guidance for class docstrings and the **Attributes** section. ([numpydoc.readthedocs.io][1])

---

## Frequent mistakes (and how to avoid them)

* **Using Google-style headings** (indented “Args”, “Returns”) while claiming NumPy style—pick one and configure Sphinx accordingly; NumPy style uses underlined headers. ([sphinx-doc.org][15])
* **Omitting types** in **Parameters/Returns** when you have annotations—NumPy style still expects types in the docstring. ([sphinx-doc.org][3])
* **Listing `self`** in methods—don’t. ([numpydoc.readthedocs.io][1])
* **Letting doctests flake**—either seed RNG or use `#random` comments to signal nondeterminism. ([numpydoc.readthedocs.io][1])
* **Bloated “Parameters”**—push rarely used kwargs to **Other Parameters**. ([numpydoc.readthedocs.io][4])
* **Ambiguous array shapes**—standardize on `(n_samples, n_features)` and related patterns. ([scikit-learn.org][6])

---

## Why the extra tooling matters

* **PEP 257** dictates the high-level behavior of docstrings (summary first line, arguments, returns, exceptions, etc.). Numpydoc refines this for scientific APIs. ([Python Enhancement Proposals (PEPs)][8])
* **`numpy.typing`** unlocks precise, checker-friendly hints (`ArrayLike`, `NDArray`, `DTypeLike`) while your docstrings stay readable for users. ([numpy.org][7])
* **Ruff/pydocstyle/numpydoc-validate** lets you *enforce* quality (ordering, capitalization, missing sections, mismatched exceptions) in CI. ([numpydoc.readthedocs.io][10])

---

### TL;DR checklist

* [ ] Short summary sentence (no parameter names)
* [ ] Sections in canonical order; underlined headers; no blank line after headers
* [ ] `Parameters` with **types**, shapes, and defaults documented (consistent phrasing)
* [ ] `Returns`/`Yields` (types required), `Raises`/`Warns`/`Warnings` as needed
* [ ] `See Also` with concise pointers
* [ ] `Notes` (optional), `References` only for durable sources
* [ ] `Examples` in doctest style; stable output or `#random` markers
* [ ] Methods exclude `self`; classes include **Attributes**
* [ ] Sphinx configured with `numpydoc` (or Napoleon if you intentionally choose it)
* [ ] Docstring linting in pre-commit/CI (numpydoc validate, pydocstyle numpy convention, Ruff DOC rules)

If you want, I can take a pass over a few of your functions/classes and return corrected numpydoc docstrings plus a ready-to-drop Sphinx + Ruff configuration.

[1]: https://numpydoc.readthedocs.io/en/latest/format.html "Style guide — numpydoc v1.10.0rc0.dev0 Manual"
[2]: https://numpy.org/doc/1.20/docs/howto_document.html?utm_source=chatgpt.com "A Guide to NumPy Documentation"
[3]: https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html?utm_source=chatgpt.com "Documentation » Example NumPy Style Python Docstrings"
[4]: https://numpydoc.readthedocs.io/en/latest/example.html?utm_source=chatgpt.com "Example — numpydoc v1.10.0rc0.dev0 Manual - Read the Docs"
[5]: https://developer.lsst.io/python/numpydoc.html?utm_source=chatgpt.com "Documenting Python APIs with docstrings"
[6]: https://scikit-learn.org/stable/developers/develop.html?utm_source=chatgpt.com "Developing scikit-learn estimators"
[7]: https://numpy.org/devdocs/reference/typing.html?utm_source=chatgpt.com "Typing (numpy.typing) — NumPy v2.4.dev0 Manual"
[8]: https://peps.python.org/pep-0257/?utm_source=chatgpt.com "PEP 257 – Docstring Conventions"
[9]: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html?utm_source=chatgpt.com "sphinx.ext.autodoc – Include documentation from docstrings"
[10]: https://numpydoc.readthedocs.io/en/latest/validation.html?utm_source=chatgpt.com "Docstring Validation using Pre-Commit Hook - numpydoc"
[11]: https://www.pydocstyle.org/_/downloads/en/6.2.1/pdf/?utm_source=chatgpt.com "pydocstyle Documentation"
[12]: https://docs.astral.sh/ruff/settings/?utm_source=chatgpt.com "Settings | Ruff - Astral Docs"
[13]: https://pypi.org/project/numpydoc-linter/?utm_source=chatgpt.com "numpydoc-linter"
[14]: https://www.pydocstyle.org/en/stable/usage.html?utm_source=chatgpt.com "Usage — pydocstyle 0.0.0.dev0 documentation"
[15]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html?utm_source=chatgpt.com "Support for NumPy and Google style docstrings"
