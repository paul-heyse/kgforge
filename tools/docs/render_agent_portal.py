"""Render the Agent Portal HTML with facets, tutorials, and feedback form."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Final

CacheKey = str

from kgfoundry.agent_catalog.client import AgentCatalogClient
from kgfoundry.agent_catalog.models import ModuleModel, SymbolModel

DEFAULT_OUTPUT = Path("site/_build/agent/index.html")
MAX_EXEMPLARS = 2
MODULE_CARD_VERSION: Final[str] = "1"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the portal renderer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("docs/_build/agent_catalog.json"),
        help="Path to the agent catalog JSON artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the rendered HTML page.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for resolving relative paths.",
    )
    return parser


def _render_quick_links(artifacts: dict[str, str]) -> str:
    entries = []
    for label, path in artifacts.items():
        if not isinstance(path, str):
            continue
        entries.append(
            f'<li><a href="{html.escape(path)}" target="_blank" rel="noopener">{html.escape(label)}</a></li>'
        )
    return "\n".join(entries) or "<li>No auxiliary artifacts published.</li>"


def _collect_module_metrics(module: ModuleModel) -> dict[str, float | int | str | None]:
    coverage_values: list[float] = []
    churn_values: list[int] = []
    stability: str | None = None
    parity_fail = False
    for symbol in module.symbols:
        if symbol.quality.docstring_coverage is not None:
            coverage_values.append(symbol.quality.docstring_coverage)
        if symbol.change_impact.churn_last_n is not None:
            churn_values.append(symbol.change_impact.churn_last_n)
        if stability is None and symbol.metrics.stability:
            stability = symbol.metrics.stability
        if symbol.quality.pydoclint_parity is False:
            parity_fail = True
    coverage = statistics.fmean(coverage_values) if coverage_values else None
    churn = max(churn_values) if churn_values else 0
    parity = "fail" if parity_fail else "pass"
    return {
        "coverage": coverage,
        "churn": churn,
        "stability": stability or "unknown",
        "parity": parity,
    }


def _collect_module_hints(symbols: Iterable[SymbolModel]) -> dict[str, list[str]]:
    tags: set[str] = set()
    safe_ops: set[str] = set()
    tests: set[str] = set()
    notes: set[str] = set()
    for symbol in symbols:
        tags.update(symbol.agent_hints.intent_tags or [])
        safe_ops.update(symbol.agent_hints.safe_ops or [])
        tests.update(symbol.agent_hints.tests_to_run or [])
        notes.update(symbol.agent_hints.breaking_change_notes or [])
    return {
        "intent_tags": sorted(tags),
        "safe_ops": sorted(safe_ops),
        "tests": sorted(tests),
        "notes": sorted(notes),
    }


def _module_cache_key(package: str, module: ModuleModel) -> CacheKey:
    """Return a stable cache key for ``module`` within ``package``."""

    payload = {
        "version": MODULE_CARD_VERSION,
        "package": package,
        "module": module.model_dump(mode="json"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_directory(output_path: Path) -> Path:
    """Return the directory used to persist cached module cards."""

    return output_path.parent / ".module_cache"


def _render_module_card_cached(
    package: str,
    module: ModuleModel,
    cache_dir: Path,
    used: set[CacheKey],
) -> str:
    """Render ``module`` using a content-addressed HTML cache when available."""

    key = _module_cache_key(package, module)
    used.add(key)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _render_module_card(package, module)
    cache_path = cache_dir / f"{key}.html"
    try:
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
    except OSError:
        pass
    markup = _render_module_card(package, module)
    try:
        cache_path.write_text(markup, encoding="utf-8")
    except OSError:
        return markup
    return markup


def _prune_cache(cache_dir: Path, used: set[CacheKey]) -> None:
    """Remove cached cards in ``cache_dir`` that were not ``used``."""

    if not cache_dir.exists():
        return
    for cached in cache_dir.glob("*.html"):
        if cached.stem not in used:
            try:
                cached.unlink(missing_ok=True)
            except OSError:
                continue


def _render_dependency_graph(module: ModuleModel) -> str:
    imports = module.graph.imports if module.graph.imports else []
    calls = [edge.get("callee") for edge in module.graph.calls or [] if isinstance(edge, dict)]
    imports_markup = (
        "<ul>" + "".join(f"<li>{html.escape(name)}</li>" for name in imports[:6]) + "</ul>"
        if imports
        else '<p class="empty">No external imports recorded.</p>'
    )
    calls_markup = (
        "<ul>" + "".join(f"<li>{html.escape(str(name))}</li>" for name in calls[:6]) + "</ul>"
        if calls
        else '<p class="empty">No call edges captured.</p>'
    )
    return (
        '<div class="graph">'
        "<h4>Dependencies</h4>"
        f"<div><strong>Imports</strong>{imports_markup}</div>"
        f"<div><strong>Calls</strong>{calls_markup}</div>"
        "</div>"
    )


def _render_exemplars(module: ModuleModel) -> str:
    exemplars = []
    for symbol in module.symbols:
        for exemplar in symbol.exemplars:
            snippet = exemplar.get("snippet", "")
            exemplars.append(
                """
                <article class="exemplar">
                  <h4>{title}</h4>
                  <pre><code>{snippet}</code></pre>
                  <button type="button" class="copy-exemplar" data-snippet="{raw}">Insert exemplar</button>
                </article>
                """.format(
                    title=html.escape(exemplar.get("title", symbol.qname)),
                    snippet=html.escape(snippet),
                    raw=html.escape(snippet, quote=True),
                )
            )
            if len(exemplars) >= MAX_EXEMPLARS:
                break
        if len(exemplars) >= MAX_EXEMPLARS:
            break
    if not exemplars:
        return '<p class="empty">No exemplars provided yet.</p>'
    return "".join(exemplars)


def _render_breadcrumbs(module_name: str) -> str:
    parts = module_name.split(".")
    items = []
    for index, part in enumerate(parts, start=1):
        items.append(
            f'<li><span aria-current="page" data-depth="{index}">{html.escape(part)}</span></li>'
        )
    return '<nav aria-label="Breadcrumb"><ol>' + "".join(items) + "</ol></nav>"


def _render_module_card(package: str, module: ModuleModel) -> str:
    metrics = _collect_module_metrics(module)
    hints = _collect_module_hints(module.symbols)
    coverage_display = (
        f"{metrics['coverage'] * 100:.0f}%" if isinstance(metrics["coverage"], float) else "—"
    )
    churn_display = str(metrics["churn"]) if isinstance(metrics["churn"], int) else "0"
    tags_markup = (
        '<ul class="pill-list">'
        + "".join(f"<li>{html.escape(tag)}</li>" for tag in hints["intent_tags"][:6])
        + "</ul>"
        if hints["intent_tags"]
        else '<p class="empty">No intent tags recorded.</p>'
    )
    tests_markup = (
        "<ul>" + "".join(f"<li>{html.escape(test)}</li>" for test in hints["tests"][:5]) + "</ul>"
        if hints["tests"]
        else '<p class="empty">No test suggestions.</p>'
    )
    docs_links = []
    html_page = module.pages.get("html")
    if html_page:
        docs_links.append(
            f'<a href="{html.escape(html_page)}" target="_blank" rel="noopener">Docs</a>'
        )
    fjson = module.pages.get("fjson")
    if fjson:
        docs_links.append(
            f'<a href="{html.escape(fjson)}" target="_blank" rel="noopener">FJSON</a>'
        )
    symbol_id = module.symbols[0].symbol_id if module.symbols else module.qualified
    docs_links.append(
        f'<button data-symbol="{html.escape(symbol_id)}" class="open-anchor" type="button">Open anchor</button>'
    )
    hints_notes = (
        "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in hints["notes"][:3]) + "</ul>"
        if hints["notes"]
        else '<p class="empty">No breaking change notes.</p>'
    )
    return f"""
    <article class="module-card" role="article"
             data-package="{html.escape(package)}"
             data-module="{html.escape(module.qualified)}"
             data-stability="{html.escape(str(metrics["stability"]))}"
             data-parity="{html.escape(str(metrics["parity"]))}"
             data-coverage="{metrics["coverage"] or 0}"
             data-churn="{metrics["churn"]}">
      {_render_breadcrumbs(module.qualified)}
      <header>
        <h3>{html.escape(module.qualified)}</h3>
        <p class="meta" aria-label="Module summary">
          <span><strong>Coverage:</strong> {coverage_display}</span>
          <span><strong>Stability:</strong> {html.escape(str(metrics["stability"]))}</span>
          <span><strong>Churn:</strong> {churn_display}</span>
          <span><strong>Parity:</strong> {html.escape(str(metrics["parity"]))}</span>
        </p>
        <p class="links" aria-label="Module resources">{" | ".join(docs_links)}</p>
      </header>
      <section>
        <h4>Agent hints</h4>
        {tags_markup}
        <div class="notes"><strong>Change impact:</strong>{hints_notes}</div>
      </section>
      <section>
        <h4>Suggested tests</h4>
        {tests_markup}
      </section>
      {_render_dependency_graph(module)}
      <section>
        <h4>Exemplars</h4>
        {_render_exemplars(module)}
      </section>
    </article>
    """


def _render_package(
    package_name: str,
    modules: list[ModuleModel],
    *,
    cache_dir: Path,
    used: set[CacheKey],
) -> str:
    cards = "\n".join(
        _render_module_card_cached(package_name, module, cache_dir, used) for module in modules
    )
    return (
        f'<section class="package" id="pkg-{html.escape(package_name)}" aria-label="Package {html.escape(package_name)}">'
        f"<h2>{html.escape(package_name)}</h2>"
        f'<div class="module-grid">{cards}</div>'
        "</section>"
    )


def _render_facets(client: AgentCatalogClient) -> str:
    """Return the HTML markup for interactive facet controls."""
    packages = sorted({package.name for package in client.list_packages()})
    stabilities = sorted(
        {
            symbol.metrics.stability or "unknown"
            for package in client.list_packages()
            for module in client.list_modules(package.name)
            for symbol in module.symbols
        }
    )
    parity_options = ["any", "pass", "fail"]
    coverage_options = [
        ("any", "Any coverage"),
        ("0.25", "25% and above"),
        ("0.5", "50% and above"),
        ("0.75", "75% and above"),
    ]
    churn_options = [
        ("any", "Any churn"),
        ("5", "Churn ≥ 5"),
        ("10", "Churn ≥ 10"),
        ("25", "Churn ≥ 25"),
    ]
    package_select = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in packages
    )
    stability_select = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value.title())}</option>'
        for value in stabilities
    )
    coverage_select = "".join(
        f'<option value="{value}">{label}</option>' for value, label in coverage_options
    )
    churn_select = "".join(
        f'<option value="{value}">{label}</option>' for value, label in churn_options
    )
    parity_select = "".join(
        f'<option value="{value}">{value.title() if value != "any" else "Any parity"}</option>'
        for value in parity_options
    )
    return f"""
    <section id="facets" aria-label="Filters">
      <form>
        <label for="facet-package">Package</label>
        <select id="facet-package">
          <option value="any">All packages</option>
          {package_select}
        </select>
        <label for="facet-stability">Stability</label>
        <select id="facet-stability">
          <option value="any">Any stability</option>
          {stability_select}
        </select>
        <label for="facet-parity">Parity</label>
        <select id="facet-parity">{parity_select}</select>
        <label for="facet-coverage">Coverage</label>
        <select id="facet-coverage">{coverage_select}</select>
        <label for="facet-churn">Churn</label>
        <select id="facet-churn">{churn_select}</select>
      </form>
    </section>
    """


def _render_tutorials() -> str:
    """Return HTML for the tutorials section."""
    tutorials = [
        ("Catalog overview", "docs/agent_portal_readme.md"),
        ("Quality checklist", "docs/contributing/quality.md"),
        ("Catalog CLI guide", "tools/agent_catalog/catalogctl.py"),
    ]
    items = "".join(
        f'<li><a href="{html.escape(path)}" target="_blank" rel="noopener">{html.escape(title)}</a></li>'
        for title, path in tutorials
    )
    return (
        '<section id="tutorials" aria-label="Tutorials and playbooks">'
        "<h2>Tutorials</h2>"
        f"<ul>{items}</ul>"
        "</section>"
    )


def _render_feedback() -> str:
    """Return HTML for the local-only feedback form."""
    return """
    <section id="feedback" aria-label="Feedback">
      <h2>Feedback</h2>
      <p>Submit local-only feedback; sensitive details are redacted automatically.</p>
      <form id="feedback-form">
        <label for="feedback-name">Name</label>
        <input id="feedback-name" name="name" type="text" autocomplete="off" />
        <label for="feedback-message">Message</label>
        <textarea id="feedback-message" name="message" rows="4" required></textarea>
        <button type="submit">Save feedback locally</button>
      </form>
      <p id="feedback-status" role="status" aria-live="polite"></p>
    </section>
    """


def render_portal(client: AgentCatalogClient, output_path: Path) -> None:
    """Render the Agent Portal HTML artifact to ``output_path``."""
    artifacts = client.catalog.artifacts
    quick_links = _render_quick_links(artifacts)
    package_sections = []
    cache_dir = _cache_directory(output_path)
    used: set[CacheKey] = set()
    for package in client.list_packages():
        modules = client.list_modules(package.name)
        package_sections.append(
            _render_package(package.name, modules, cache_dir=cache_dir, used=used)
        )
    packages_markup = "\n".join(package_sections)
    facets_markup = _render_facets(client)
    tutorials_markup = _render_tutorials()
    feedback_markup = _render_feedback()
    html_output = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Agent Portal</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #0f172a;
        --surface: #1e293b;
        --accent: #38bdf8;
        --muted: #94a3b8;
      }}
      * {{ box-sizing: border-box; }}
      body {{ font-family: system-ui, sans-serif; margin: 0; background: var(--bg); color: #e2e8f0; }}
      header {{ background: var(--surface); padding: 1.5rem; position: sticky; top: 0; z-index: 2; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }}
      .sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }}
      header h1 {{ margin: 0 0 0.5rem; font-size: 1.6rem; }}
      header p {{ margin: 0.25rem 0; color: var(--muted); }}
      #search {{ width: 100%; max-width: 420px; padding: 0.75rem; border-radius: 0.75rem; border: 1px solid #334155; background: #0b1120; color: inherit; }}
      main {{ padding: 1.5rem; display: grid; gap: 2rem; grid-template-columns: minmax(0, 1fr); }}
      #facets form {{ display: grid; gap: 0.5rem; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
      #facets select, #facets input {{ padding: 0.5rem; border-radius: 0.5rem; border: 1px solid #334155; background: #0b1120; color: inherit; }}
      .quick-links ul {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 1rem; }}
      .quick-links a {{ color: var(--accent); text-decoration: none; }}
      .quick-links a:focus-visible {{ outline: 2px solid #fbbf24; outline-offset: 2px; }}
      .module-grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
      .module-card {{ background: var(--surface); padding: 1.25rem; border-radius: 1rem; display: grid; gap: 1rem; grid-template-columns: minmax(0, 1fr); box-shadow: 0 15px 35px rgba(8,11,19,0.6); transition: transform 0.2s ease; }}
      .module-card:focus-within, .module-card:hover {{ transform: translateY(-3px); }}
      .module-card header {{ display: grid; gap: 0.5rem; }}
      .module-card nav ol {{ list-style: none; padding: 0; margin: 0; display: flex; gap: 0.35rem; font-size: 0.85rem; color: var(--muted); }}
      .module-card nav li::after {{ content: '\\203A'; margin-left: 0.35rem; }}
      .module-card nav li:last-child::after {{ content: ''; }}
      .module-card .meta span {{ margin-right: 0.75rem; display: inline-block; }}
      .module-card .links {{ display: flex; gap: 0.75rem; flex-wrap: wrap; }}
      .module-card .links a, .module-card .links button {{ background: none; border: none; color: var(--accent); cursor: pointer; padding: 0; font: inherit; }}
      .module-card .links button:hover, .module-card .links button:focus-visible {{ text-decoration: underline; outline: none; }}
      .pill-list {{ list-style: none; padding: 0; display: flex; gap: 0.5rem; flex-wrap: wrap; }}
      .pill-list li {{ background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.4); border-radius: 999px; padding: 0.25rem 0.75rem; font-size: 0.85rem; }}
      .graph {{ border-top: 1px solid #334155; padding-top: 0.75rem; display: grid; gap: 0.5rem; }}
      .graph h4 {{ margin: 0; font-size: 1rem; }}
      .graph ul {{ list-style: none; padding-left: 0; margin: 0; font-size: 0.9rem; color: var(--muted); }}
      .graph div {{ display: grid; gap: 0.25rem; }}
      .exemplar {{ border: 1px solid #334155; border-radius: 0.75rem; padding: 0.75rem; background: #0b1120; }}
      .exemplar pre {{ margin: 0; overflow-x: auto; }}
      .empty {{ color: var(--muted); font-style: italic; }}
      footer {{ background: var(--surface); padding: 1.5rem; }}
      #feedback form {{ display: grid; gap: 0.75rem; max-width: 520px; }}
      #feedback input, #feedback textarea {{ padding: 0.6rem; border-radius: 0.5rem; border: 1px solid #334155; background: #0b1120; color: inherit; }}
      #feedback button {{ background: var(--accent); border: none; color: #0b1120; padding: 0.75rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; }}
      #feedback button:hover, #feedback button:focus-visible {{ filter: brightness(1.1); outline: none; }}
      #results-count {{ margin-top: 1rem; color: var(--muted); }}
      @media (max-width: 720px) {{
        header {{ position: static; }}
        #facets form {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <header role="banner">
      <h1>Agent Portal</h1>
      <p>Explore packages, drill into symbols, and collect change impact evidence.</p>
      <label for="search" class="sr-only">Search modules and packages</label>
      <input id="search" type="search" placeholder="Search modules and packages" aria-label="Search" />
      <div class="quick-links" role="navigation" aria-label="Artifact shortcuts">
        <strong>Artifacts:</strong>
        <ul>{quick_links}</ul>
      </div>
      <noscript>Interactive facets and search require JavaScript. All modules are listed below.</noscript>
    </header>
    <main role="main">
      {facets_markup}
      <section aria-label="Results">
        <div id="results-count" aria-live="polite" role="status">Displaying all modules</div>
        <div id="packages">{packages_markup}</div>
      </section>
      {tutorials_markup}
    </main>
    <footer>
      {feedback_markup}
    </footer>
    <script>
      const searchInput = document.getElementById('search');
      const packageNodes = Array.from(document.querySelectorAll('.package'));
      const moduleCards = Array.from(document.querySelectorAll('.module-card'));
      const packageFacet = document.getElementById('facet-package');
      const stabilityFacet = document.getElementById('facet-stability');
      const parityFacet = document.getElementById('facet-parity');
      const coverageFacet = document.getElementById('facet-coverage');
      const churnFacet = document.getElementById('facet-churn');
      const resultsCount = document.getElementById('results-count');

      function numeric(value) {{
        const parsed = parseFloat(value);
        return Number.isNaN(parsed) ? 0 : parsed;
      }}

      function matchesFacets(card, term) {{
        const text = (card.dataset.module + ' ' + card.dataset.package).toLowerCase();
        if (term && !text.includes(term)) {{
          return false;
        }}
        if (packageFacet.value !== 'any' && card.dataset.package !== packageFacet.value) {{
          return false;
        }}
        if (stabilityFacet.value !== 'any' && card.dataset.stability !== stabilityFacet.value) {{
          return false;
        }}
        if (parityFacet.value !== 'any' && card.dataset.parity !== parityFacet.value) {{
          return false;
        }}
        if (coverageFacet.value !== 'any' && numeric(card.dataset.coverage) < numeric(coverageFacet.value)) {{
          return false;
        }}
        if (churnFacet.value !== 'any' && numeric(card.dataset.churn) < numeric(churnFacet.value)) {{
          return false;
        }}
        return true;
      }}

      function updateVisibility() {{
        const term = searchInput.value.trim().toLowerCase();
        let visible = 0;
        moduleCards.forEach((card) => {{
          const show = matchesFacets(card, term);
          card.style.display = show ? '' : 'none';
          if (show) {{ visible += 1; }}
        }});
        packageNodes.forEach((pkg) => {{
          const hasVisible = Array.from(pkg.querySelectorAll('.module-card')).some((card) => card.style.display !== 'none');
          pkg.style.display = hasVisible ? '' : 'none';
        }});
        resultsCount.textContent = visible === moduleCards.length ? 'Displaying all modules' : 'Displaying ' + visible + ' modules';
      }}

      [searchInput, packageFacet, stabilityFacet, parityFacet, coverageFacet, churnFacet].forEach((control) => {{
        control?.addEventListener('input', updateVisibility);
        control?.addEventListener('change', updateVisibility);
      }});

      document.querySelectorAll('.open-anchor').forEach((button) => {{
        button.addEventListener('click', () => {{
          const symbolId = button.getAttribute('data-symbol');
          if (!symbolId) return;
          fetch('catalogctl://open?symbol_id=' + encodeURIComponent(symbolId)).catch(() => {{}});
        }});
      }});

      document.querySelectorAll('.copy-exemplar').forEach((button) => {{
        button.addEventListener('click', async () => {{
          const snippet = button.getAttribute('data-snippet') || '';
          try {{
            await navigator.clipboard.writeText(snippet);
            button.textContent = 'Copied!';
            setTimeout(() => (button.textContent = 'Insert exemplar'), 1500);
          }} catch (error) {{
            console.warn('Clipboard unavailable', error);
          }}
        }});
      }});

      const feedbackForm = document.getElementById('feedback-form');
      const feedbackStatus = document.getElementById('feedback-status');

      function sanitizeFeedback(text) {{
        return text
          .replace(/[a-z0-9._%+-]+@[a-z0-9.-]+/gi, '[redacted-email]')
          .replace(/\b\\d{{4,}}\b/g, '[redacted-number]');
      }}

      feedbackForm?.addEventListener('submit', (event) => {{
        event.preventDefault();
        const nameInput = document.getElementById('feedback-name');
        const messageInput = document.getElementById('feedback-message');
        const name = nameInput instanceof HTMLInputElement ? nameInput.value : '';
        const message = messageInput instanceof HTMLTextAreaElement ? messageInput.value : '';
        const sanitized = {{ name: sanitizeFeedback(name), message: sanitizeFeedback(message), submitted_at: new Date().toISOString() }};
        try {{
          const existing = JSON.parse(localStorage.getItem('agent-portal-feedback') || '[]');
          existing.push(sanitized);
          localStorage.setItem('agent-portal-feedback', JSON.stringify(existing));
        }} catch (error) {{
          console.warn('Unable to persist feedback to localStorage', error);
        }}
        const blob = new Blob([JSON.stringify(sanitized, null, 2)], {{ type: 'application/json' }});
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = 'agent-feedback-' + Date.now() + '.json';
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
        if (feedbackStatus) {{
          feedbackStatus.textContent = 'Feedback stored locally. JSON download triggered.';
        }}
        if (nameInput instanceof HTMLInputElement) nameInput.value = '';
        if (messageInput instanceof HTMLTextAreaElement) messageInput.value = '';
      }});

      updateVisibility();
    </script>
  </body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_output, encoding="utf-8")
    _prune_cache(cache_dir, used)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = AgentCatalogClient.from_path(args.catalog, repo_root=args.repo_root)
    render_portal(client, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
