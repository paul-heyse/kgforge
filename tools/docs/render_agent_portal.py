"""Render the Agent Portal HTML using the catalog artifacts."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from kgfoundry.agent_catalog.client import AgentCatalogClient

DEFAULT_OUTPUT = Path("site/_build/agent/index.html")


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
    """Return HTML for quick links referencing other artifacts."""
    entries = []
    for label, path in artifacts.items():
        if not isinstance(path, str):
            continue
        href = html.escape(path)
        entries.append(f'<li><a href="{href}">{html.escape(label)}</a></li>')
    return "\n".join(entries)


def _render_module_card(client: AgentCatalogClient, package: str, module_name: str) -> str:
    """Return HTML for a module summary card."""
    module = client.get_module(module_name)
    if module is None:
        return ""
    page_html = module.pages.get("html") or ""
    fjson = module.pages.get("fjson") or ""
    symbol_count = len(module.symbols)
    anchors = []
    for symbol in module.symbols[:6]:
        anchor = (
            f"<li><code>{html.escape(symbol.qname)}</code> "
            f'<span class="metric">{len(symbol.change_impact.callers)} callers</span></li>'
        )
        anchors.append(anchor)
    symbols_markup = "\n".join(anchors) or "<li>No symbols indexed.</li>"
    open_links = []
    if page_html:
        open_links.append(f'<a href="{html.escape(page_html)}">Docs</a>')
    if fjson:
        open_links.append(f'<a href="{html.escape(fjson)}">FJSON</a>')
    symbol_id = module.symbols[0].symbol_id if module.symbols else module.qualified
    open_links.append(
        f'<a data-action="open" data-symbol="{html.escape(symbol_id)}" href="#">Open</a>'
    )
    links_markup = " | ".join(open_links)
    return (
        f'<section class="module" data-package="{html.escape(package)}" '
        f'data-module="{html.escape(module.qualified)}">'
        f"<h3>{html.escape(module.qualified)}</h3>"
        f'<p class="meta">{symbol_count} symbols · {html.escape(package)}</p>'
        f'<p class="links">{links_markup}</p>'
        f'<ul class="symbols">{symbols_markup}</ul>'
        "</section>"
    )


def _render_package_section(client: AgentCatalogClient, package: str) -> str:
    """Return HTML for a package section containing modules."""
    modules_markup = "\n".join(
        _render_module_card(client, package, module.qualified)
        for module in client.list_modules(package)
    )
    return (
        f'<section class="package" id="pkg-{html.escape(package)}">'
        f"<h2>{html.escape(package)}</h2>"
        f'<div class="modules">{modules_markup}</div>'
        "</section>"
    )


def render_portal(client: AgentCatalogClient, output_path: Path) -> None:
    """Render the Agent Portal HTML to ``output_path``."""
    artifacts = client.catalog.artifacts
    quick_links = _render_quick_links(artifacts)
    packages_markup = "\n".join(
        _render_package_section(client, package.name) for package in client.list_packages()
    )
    html_output = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Agent Portal</title>
    <style>
      body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }}
      header {{ background: #1e293b; padding: 1.5rem; position: sticky; top: 0; z-index: 1; }}
      header h1 {{ margin: 0 0 0.5rem 0; font-size: 1.5rem; }}
      #search {{ width: 100%; max-width: 480px; padding: 0.6rem; border-radius: 0.5rem; border: none; }}
      main {{ padding: 1.5rem; }}
      .quick-links ul {{ list-style: none; padding: 0; display: flex; gap: 1rem; flex-wrap: wrap; }}
      .quick-links a {{ color: #38bdf8; text-decoration: none; }}
      .package {{ margin-bottom: 2rem; }}
      .package h2 {{ border-bottom: 1px solid #334155; padding-bottom: 0.5rem; }}
      .modules {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
      .module {{ background: #1e293b; padding: 1rem; border-radius: 0.75rem; box-shadow: 0 10px 30px rgba(15,23,42,0.45); }}
      .module h3 {{ margin-top: 0; font-size: 1.1rem; }}
      .module .meta {{ color: #94a3b8; margin-bottom: 0.5rem; }}
      .module .links a {{ color: #38bdf8; text-decoration: none; margin-right: 0.5rem; }}
      .module ul {{ list-style: none; padding: 0; margin: 0; }}
      .module li {{ margin-bottom: 0.3rem; }}
      .module code {{ background: #0f172a; padding: 0.1rem 0.3rem; border-radius: 0.3rem; }}
      .module .metric {{ color: #94a3b8; margin-left: 0.5rem; font-size: 0.85rem; }}
      noscript {{ display: block; margin-top: 1rem; color: #fbbf24; }}
    </style>
  </head>
  <body>
    <header>
      <h1>Agent Portal</h1>
      <input id=\"search\" type=\"search\" placeholder=\"Search modules and packages\" aria-label=\"Search\" />
      <div class=\"quick-links\">
        <strong>Artifacts:</strong>
        <ul>{quick_links}</ul>
      </div>
      <noscript>Search and filtering require JavaScript. All packages are listed below.</noscript>
    </header>
    <main>
      <div id=\"packages\">{packages_markup}</div>
    </main>
    <script>
      const packageSections = Array.from(document.querySelectorAll('.package'));
      const modules = Array.from(document.querySelectorAll('.module'));
      const searchInput = document.getElementById('search');
      function normalize(text) {{ return text.toLowerCase(); }}
      function filter(value) {{
        const term = normalize(value.trim());
        modules.forEach((module) => {{
          const matches = !term || normalize(module.dataset.module).includes(term) || normalize(module.dataset.package).includes(term);
          module.style.display = matches ? '' : 'none';
        }});
        packageSections.forEach((pkg) => {{
          const visible = Array.from(pkg.querySelectorAll('.module')).some((node) => node.style.display !== 'none');
          pkg.style.display = visible ? '' : 'none';
        }});
      }}
      searchInput.addEventListener('input', (event) => filter(event.target.value));
    </script>
  </body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_output, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the portal renderer CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    client = AgentCatalogClient.from_path(args.catalog, repo_root=args.repo_root)
    render_portal(client, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
