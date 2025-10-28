#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Third-party (global graph & rendering)
try:
    import pydot  # DOT <-> pydot
except Exception:
    pydot = None  # handled in main()
try:
    import networkx as nx  # cycles, centrality
except Exception:
    nx = None  # handled in main()
try:
    import yaml  # layers policy
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

# Defaults / policy files
LAYER_FILE = ROOT / "docs" / "policies" / "layers.yml"
ALLOW_FILE = ROOT / "docs" / "policies" / "graph_allowlist.json"

# CLI -------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-package and cross-subsystem graphs with policy checks."
    )
    p.add_argument(
        "--packages",
        default=os.getenv("DOCS_PKG", ""),
        help="Comma-separated list of top-level packages to include (default: auto-detect under src/)",
    )
    p.add_argument(
        "--format",
        default=os.getenv("GRAPH_FORMAT", "svg"),
        choices=["svg", "png"],
        help="Image format for rendered graphs",
    )
    p.add_argument(
        "--max-bacon",
        type=int,
        default=int(os.getenv("GRAPH_MAX_BACON", "4")),
        help="pydeps --max-bacon (default: 4)",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Regex/pattern to exclude in pydeps (repeatable)",
    )
    p.add_argument(
        "--layers",
        default=str(LAYER_FILE),
        help="Path to layers.yml (layer names & package mapping)",
    )
    p.add_argument(
        "--allowlist", default=str(ALLOW_FILE), help="Path to graph allowlist (cycles/edges)"
    )
    p.add_argument(
        "--fail-on-cycles",
        action="store_true",
        default=os.getenv("GRAPH_FAIL_ON_CYCLES", "1") == "1",
        help="Fail when new cycles (not allowlisted) are found (default: on)",
    )
    p.add_argument(
        "--fail-on-layer-violations",
        action="store_true",
        default=os.getenv("GRAPH_FAIL_ON_LAYER", "1") == "1",
        help="Fail on forbidden cross-layer edges (default: on)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


# Utilities -------------------------------------------------------------------


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, check=check, cwd=str(cwd) if cwd else None, text=True, capture_output=False
    )


def ensure_bin(name: str) -> None:
    if not shutil.which(name):
        print(f"[graphs] Missing required executable on PATH: {name}", file=sys.stderr)
        sys.exit(2)


def find_top_packages() -> list[str]:
    # Top-level packages are directories under src/ that contain __init__.py
    pkgs: list[str] = []
    if not SRC.exists():
        return pkgs
    for child in SRC.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            pkgs.append(child.name)
    return sorted(pkgs)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


# Per-package rendering --------------------------------------------------------


def build_pydeps_for_package(
    pkg: str, out_svg: Path, excludes: list[str], max_bacon: int, fmt: str
) -> None:
    # pydeps <module> --noshow --show-dot --dot-output ... -T dot [excludes] --max-bacon N
    dot_tmp = out_svg.with_suffix(".dot")
    cmd = [
        sys.executable,
        "-m",
        "pydeps",
        f"src/{pkg}",
        "--noshow",
        "--show-dot",
        "--dot-output",
        str(dot_tmp),
        "--max-bacon",
        str(max_bacon),
        "-T",
        "dot",
    ]
    for pat in excludes:
        cmd += ["-x", pat]
    sh(cmd, cwd=ROOT)
    # Render to final image via graphviz (dot)
    sh(["dot", f"-T{fmt}", str(dot_tmp), "-o", str(out_svg)], cwd=ROOT)


def build_pyreverse_for_package(pkg: str, out_dir: Path, fmt: str) -> None:
    # pyreverse <pkg> -o svg -p <pkg> (Graphviz must be installed)
    # We render classes_*.svg -> <pkg>-uml.svg
    sh(["pyreverse", f"src/{pkg}", "-o", fmt, "-p", pkg], cwd=ROOT)
    # Move/rename the class diagram (pyreverse emits classes_*.svg)
    for svg in ROOT.glob("classes_*.svg"):
        svg.replace(out_dir / f"{pkg}-uml.{fmt}")


# Global graph (collapsed to packages) ----------------------------------------


def _pkg_of(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def build_global_pydeps(dot_out: Path, excludes: list[str], max_bacon: int) -> None:
    # pydeps src --noshow --show-dot --dot-output subsystems.dot --max-bacon N
    cmd = [
        sys.executable,
        "-m",
        "pydeps",
        "src",
        "--noshow",
        "--show-dot",
        "--dot-output",
        str(dot_out),
        "--max-bacon",
        str(max_bacon),
        "-T",
        "dot",
    ]
    for pat in excludes:
        cmd += ["-x", pat]
    sh(cmd, cwd=ROOT)


def collapse_to_packages(dot_path: Path):
    graphs = pydot.graph_from_dot_file(str(dot_path))
    pd = graphs[0] if isinstance(graphs, list) else graphs
    g = nx.drawing.nx_pydot.from_pydot(pd).to_directed()
    collapsed = nx.DiGraph()
    for u, v in g.edges():
        pu, pv = _pkg_of(str(u)), _pkg_of(str(v))
        if pu != pv:
            w = collapsed.get_edge_data(pu, pv, {}).get("weight", 0) + 1
            collapsed.add_edge(pu, pv, weight=w)
    return collapsed


def analyze_graph(g, layers: dict[str, Any]) -> dict[str, Any]:
    # cycles (Johnson’s algorithm) & degree centrality
    cycles = [c for c in nx.simple_cycles(g)]  # Johnson’s algo; see docs
    centrality = nx.degree_centrality(g) if g.number_of_nodes() else {}
    # layer policy violations (dependencies should point inward in the layer order)
    order = layers.get("order", [])
    pkg2layer: dict[str, str] = layers.get("packages", {}) or {}
    rank = {name: i for i, name in enumerate(order)}
    violations: list[list[str]] = []
    rules = layers.get("rules", {})
    allow_outward = bool(rules.get("allow_outward", False))
    for u, v in g.edges():
        lu, lv = pkg2layer.get(u), pkg2layer.get(v)
        if lu and lv and lu in rank and lv in rank:
            # outward = toward a greater rank index
            if rank[lv] > rank[lu] and not allow_outward:
                violations.append([lu, lv, f"edge:{u}->{v}"])
    return {"cycles": cycles, "centrality": centrality, "layer_violations": violations}


def style_and_render(
    g, layers: dict[str, Any], analysis: dict[str, Any], out_svg: Path, fmt: str = "svg"
) -> None:
    pkg2layer = layers.get("packages", {}) or {}
    # palette for layers
    palette = {
        "domain": "#2f855a",
        "application": "#3182ce",
        "interface": "#805ad5",
        "infra": "#dd6b20",
    }

    # centrality threshold (highlight heavy nodes)
    cent = analysis["centrality"] or {}
    values = list(cent.values())
    q80 = sorted(values)[int(0.80 * len(values))] if values else 0.0

    # cycle edges set
    cycle_edges = set()
    for cyc in analysis["cycles"]:
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            cycle_edges.add((u, v))

    # pydot graph (rank left-to-right)
    pd = pydot.Dot(graph_type="digraph", rankdir="LR")

    # nodes
    for n in g.nodes():
        layer = pkg2layer.get(n, "unknown")
        color = palette.get(layer, "#718096")
        width = 2.5 if cent.get(n, 0) >= q80 else 1.2
        pd.add_node(
            pydot.Node(n, color=color, penwidth=str(width), style="bold", fontcolor="#1a202c")
        )

    # edges
    for u, v, data in g.edges(data=True):
        layer_u = pkg2layer.get(u, "unknown")
        layer_v = pkg2layer.get(v, "unknown")
        edge_color = "#e53e3e" if (u, v) in cycle_edges else "#a0aec0"  # red for cycles
        penw = "2.5" if (u, v) in cycle_edges else "1.2"
        label = f"{layer_u}→{layer_v}" if layer_u != layer_v else ""
        pd.add_edge(pydot.Edge(u, v, color=edge_color, penwidth=penw, fontsize="8", label=label))

    # Render
    if fmt not in {"svg", "png"}:
        fmt = "svg"
    data = pd.create_svg() if fmt == "svg" else pd.create_png()
    out_svg.write_bytes(data)


def write_meta(meta: dict[str, Any], out_json: Path) -> None:
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def enforce_policy(
    analysis: dict[str, Any], allow: dict[str, Any], fail_cycles: bool, fail_layers: bool
) -> None:
    allowed_cycles = set(tuple(c) for c in (allow.get("cycles") or []))
    allowed_edges = set(tuple(e) for e in (allow.get("edges") or []))

    new_cycles = [c for c in analysis["cycles"] if tuple(c) not in allowed_cycles]
    new_viol = [
        v
        for v in analysis["layer_violations"]
        if tuple(v[-1].split(":")[-1].split("->")) not in allowed_edges
    ]

    errs = []
    if fail_cycles and new_cycles:
        errs.append(f"{len(new_cycles)} cycle(s) not allowlisted")
    if fail_layers and new_viol:
        errs.append(f"{len(new_viol)} layer violation edge(s) not allowlisted")
    if errs:
        print("[graphs] policy violations:", "; ".join(errs))
        sys.exit(2)


# Main ------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Lazy imports check for global graph path
    missing = []
    if pydot is None:
        missing.append("pydot")
    if nx is None:
        missing.append("networkx")
    if yaml is None:
        missing.append("pyyaml")
    if missing:
        print(
            f"[graphs] Missing Python packages: {', '.join(missing)}. Install them in the docs env.",
            file=sys.stderr,
        )
        sys.exit(2)

    # graphviz 'dot' is required for rendering both pydeps (to svg/png) and pyreverse
    if not shutil.which("dot"):
        print("[graphs] graphviz 'dot' not found on PATH. Install graphviz.", file=sys.stderr)
        sys.exit(2)

    packages: list[str] = (
        [s.strip() for s in args.packages.split(",") if s.strip()]
        if args.packages
        else find_top_packages()
    )
    if args.verbose:
        print(f"[graphs] packages={packages}")

    excludes = args.exclude or ["tests/.*", "site/.*"]

    # 1) Per-package graphs
    t0 = time.time()
    for pkg in packages:
        try:
            build_pydeps_for_package(
                pkg, OUT / f"{pkg}-imports.{args.format}", excludes, args.max_bacon, args.format
            )
        except subprocess.CalledProcessError:
            print(f"[graphs] pydeps failed for {pkg} (continuing)", file=sys.stderr)
        try:
            build_pyreverse_for_package(pkg, OUT, args.format)
        except subprocess.CalledProcessError:
            print(f"[graphs] pyreverse failed for {pkg} (continuing)", file=sys.stderr)

    # 2) Global collapsed graph (subsystems)
    dot_all = OUT / "subsystems.dot"
    try:
        build_global_pydeps(dot_all, excludes, args.max_bacon)
        g = collapse_to_packages(dot_all)
    except Exception as e:
        print(f"[graphs] building global graph failed: {e}", file=sys.stderr)
        sys.exit(2)

    # 3) Load policy & allowlist
    layers = (
        yaml.safe_load(Path(args.layers).read_text())
        if Path(args.layers).exists()
        else {"order": [], "packages": {}, "rules": {}}
    )
    allow = (
        json.loads(Path(args.allowlist).read_text())
        if Path(args.allowlist).exists()
        else {"cycles": [], "edges": []}
    )

    # 4) Analyze, render, write meta
    analysis = analyze_graph(g, layers)
    style_and_render(g, layers, analysis, OUT / f"subsystems.{args.format}", fmt=args.format)
    meta = {
        "packages": sorted([str(n) for n in g.nodes()]),
        "cycles": analysis["cycles"],
        "centrality": analysis["centrality"],
        "layer_violations": analysis["layer_violations"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_meta(meta, OUT / "graph_meta.json")

    # 5) Enforce policy
    enforce_policy(analysis, allow, args.fail_on_cycles, args.fail_on_layer_violations)

    if args.verbose:
        print(f"[graphs] done in {time.time() - t0:.2f}s; outputs in {OUT}")


if __name__ == "__main__":
    import shutil

    main()
