#!/usr/bin/env python3
"""Overview of build graphs.

This module bundles build graphs logic for the kgfoundry stack. It groups related helpers so
downstream packages can import a single cohesive namespace. Refer to the functions and classes below
for implementation specifics.
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

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

# Render targets supported by both Graphviz and our docs pipeline.
SUPPORTED_FORMATS: tuple[str, ...] = ("svg", "png")

# Defaults / policy files
LAYER_FILE = ROOT / "docs" / "policies" / "layers.yml"
ALLOW_FILE = ROOT / "docs" / "policies" / "graph_allowlist.json"

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Compute parse args.

    Carry out the parse args operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Returns
    -------
    argparse.Namespace
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import parse_args
    >>> result = parse_args()
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    p = argparse.ArgumentParser(
        description="Build per-package and cross-subsystem graphs with policy checks."
    )
    p.add_argument(
        "--packages",
        default=os.getenv("DOCS_PKG", ""),
        help="Comma-separated top-level packages (default: auto-detect under src/)",
    )
    p.add_argument(
        "--format",
        default=os.getenv("GRAPH_FORMAT", "svg"),
        choices=list(SUPPORTED_FORMATS),
        help="Image format for rendered graphs (svg or png)",
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
    # NEW: parallel + cache knobs
    p.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("GRAPH_MAX_WORKERS", "0")),  # 0 => auto (cpu_count)
        help="Maximum parallel workers for per-package builds (default: CPU count)",
    )
    p.add_argument(
        "--cache-dir",
        default=os.getenv("GRAPH_CACHE_DIR", str(ROOT / ".cache" / "graphs")),
        help="Directory for per-package graph cache",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (always rebuild per-package graphs)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Compute sh.

    Carry out the sh operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    cmd : List[str]
        Description for ``cmd``.
    cwd : Path | None
        Description for ``cwd``.
    check : bool | None
        Description for ``check``.
    
    Returns
    -------
    subprocess.CompletedProcess
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import sh
    >>> result = sh(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    return subprocess.run(
        cmd, check=check, cwd=str(cwd) if cwd else None, text=True, capture_output=False
    )


def ensure_bin(name: str) -> None:
    """Compute ensure bin.

    Carry out the ensure bin operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    name : str
        Description for ``name``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import ensure_bin
    >>> ensure_bin(...)  # doctest: +ELLIPSIS
    """
    
    if not shutil.which(name):
        print(f"[graphs] Missing required executable on PATH: {name}", file=sys.stderr)
        print(
            "[graphs] Run 'uv sync --frozen --extra docs' (or './scripts/bootstrap.sh') to install it.",
            file=sys.stderr,
        )
        sys.exit(2)


def find_top_packages() -> list[str]:
    """Compute find top packages.

    Carry out the find top packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Returns
    -------
    List[str]
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import find_top_packages
    >>> result = find_top_packages()
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    # Top-level packages are directories under src/ that contain __init__.py
    pkgs: list[str] = []
    if not SRC.exists():
        return pkgs
    for child in SRC.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            pkgs.append(child.name)
    return sorted(pkgs)


def _rel(p: Path) -> str:
    """Return ``p`` relative to the repository root when possible."""
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


# --------------------------------------------------------------------------------------
# Per-package renderers (pydeps & pyreverse)
# --------------------------------------------------------------------------------------


def build_pydeps_for_package(
    pkg: str, out_svg: Path, excludes: list[str], max_bacon: int, fmt: str
) -> None:
    """Compute build pydeps for package.

    Carry out the build pydeps for package operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    pkg : str
        Description for ``pkg``.
    out_svg : Path
        Description for ``out_svg``.
    excludes : List[str]
        Description for ``excludes``.
    max_bacon : int
        Description for ``max_bacon``.
    fmt : str
        Description for ``fmt``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import build_pydeps_for_package
    >>> build_pydeps_for_package(..., ..., ..., ..., ...)  # doctest: +ELLIPSIS
    """
    
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
    if dot_tmp.exists():
        dot_tmp.unlink()


def build_pyreverse_for_package(pkg: str, out_dir: Path, fmt: str) -> None:
    """Compute build pyreverse for package.

    Carry out the build pyreverse for package operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    pkg : str
        Description for ``pkg``.
    out_dir : Path
        Description for ``out_dir``.
    fmt : str
        Description for ``fmt``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import build_pyreverse_for_package
    >>> build_pyreverse_for_package(..., ..., ...)  # doctest: +ELLIPSIS
    """
    
    # classes_<project>.dot is named by -p <project>; use the package name to get unique names.
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(
        [
            "pyreverse",
            f"src/{pkg}",
            "-o",
            "dot",
            "-p",
            pkg,
            "-d",
            str(out_dir),
        ],
        cwd=ROOT,
    )

    classes_dot = out_dir / f"classes_{pkg}.dot"
    packages_dot = out_dir / f"packages_{pkg}.dot"
    output_path = out_dir / f"{pkg}-uml.{fmt}"
    if classes_dot.exists():
        sh(["dot", f"-T{fmt}", str(classes_dot), "-o", str(output_path)], cwd=ROOT)

    # Cleanup intermediate dot files from both the build directory and repo root
    for dot_path in (classes_dot, packages_dot):
        if dot_path.exists():
            dot_path.unlink()

    for root_dot in (
        ROOT / f"classes_{pkg}.dot",
        ROOT / f"packages_{pkg}.dot",
    ):
        if root_dot.exists():
            root_dot.unlink()


# --------------------------------------------------------------------------------------
# Global graph (collapsed to packages)
# --------------------------------------------------------------------------------------


def _pkg_of(dotted: str) -> str:
    """Return the top-level package segment from a dotted module path."""
    head, *_rest = dotted.split(".", 1)
    if head == "src" and "." in dotted:
        return dotted.split(".", 2)[1]
    return head


def build_global_pydeps(dot_out: Path, excludes: list[str], max_bacon: int) -> None:
    """Compute build global pydeps.

    Carry out the build global pydeps operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    dot_out : Path
        Description for ``dot_out``.
    excludes : List[str]
        Description for ``excludes``.
    max_bacon : int
        Description for ``max_bacon``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import build_global_pydeps
    >>> build_global_pydeps(..., ..., ...)  # doctest: +ELLIPSIS
    """
    
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
    # optional: limit depth & drop hubs
    noise = os.getenv("GRAPH_NOISE_LEVEL")
    if noise:
        cmd += ["--noise-level", noise]
    mdepth = os.getenv("GRAPH_MAX_MODULE_DEPTH")
    if mdepth:
        cmd += ["--max-module-depth", mdepth]
    for pat in excludes:
        cmd += ["-x", pat]
    sh(cmd, cwd=ROOT)


def collapse_to_packages(dot_path: Path) -> nx.DiGraph:
    """Compute collapse to packages.

    Carry out the collapse to packages operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    dot_path : Path
        Description for ``dot_path``.
    
    Returns
    -------
    nx.DiGraph
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import collapse_to_packages
    >>> result = collapse_to_packages(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    graphs = pydot.graph_from_dot_file(str(dot_path))
    pd = graphs[0] if isinstance(graphs, list) else graphs
    g = nx.drawing.nx_pydot.from_pydot(pd).to_directed()
    collapsed = nx.DiGraph()

    def _module_name(node: Any, data: dict[str, Any]) -> str:
        label = data.get("label")
        if label:
            raw = str(label).strip("\"")
            normalized = raw.replace(r"\n", "\n").replace(r"\.", ".").replace(r"\\", "\\")
            return normalized.replace("\n", "")
        return str(node)

    module_names = {node: _module_name(node, data) for node, data in g.nodes(data=True)}

    package_nodes = {_pkg_of(name) for name in module_names.values()}
    collapsed.add_nodes_from(sorted(package_nodes))

    for u, v in g.edges():
        pu, pv = _pkg_of(module_names.get(u, str(u))), _pkg_of(module_names.get(v, str(v)))
        if pu != pv:
            w = collapsed.get_edge_data(pu, pv, {}).get("weight", 0) + 1
            collapsed.add_edge(pu, pv, weight=w)
    return collapsed


def analyze_graph(g: nx.DiGraph, layers: dict[str, Any]) -> dict[str, Any]:
    """Compute analyze graph.

    Carry out the analyze graph operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    g : nx.DiGraph
        Description for ``g``.
    layers : collections.abc.Mapping
        Description for ``layers``.
    
    Returns
    -------
    collections.abc.Mapping
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import analyze_graph
    >>> result = analyze_graph(..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    # cycles (Johnson's algorithm) & degree centrality
    # 1) prune forbidden outward edges before cycle enumeration
    order = layers.get("order", [])
    pkg2layer: dict[str, str] = layers.get("packages", {}) or {}
    rank = {name: i for i, name in enumerate(order)}
    rules = layers.get("rules", {})
    allow_outward = bool(rules.get("allow_outward", False))
    gp = g.copy()
    to_drop = []
    for u, v in g.edges():
        lu, lv = pkg2layer.get(u), pkg2layer.get(v)
        if lu and lv and lu in rank and lv in rank and rank[lv] > rank[lu] and not allow_outward:
            to_drop.append((u, v))
    if to_drop:
        gp.remove_edges_from(to_drop)

    # Prefer bounded enumeration if available (NetworkX >= 3.5)
    cycle_limit = int(os.getenv("GRAPH_CYCLE_LIMIT", "0"))  # 0 = unbounded stream
    length_bound = int(os.getenv("GRAPH_CYCLE_LEN", "0"))  # 0 = unbounded
    cycles = []
    try:
        if length_bound > 0:
            cyc_iter = nx.simple_cycles(gp, length_bound=length_bound)
        else:
            cyc_iter = nx.simple_cycles(gp)
    except TypeError:
        # older NetworkX without length_bound
        cyc_iter = nx.simple_cycles(gp)

    cycle_enumeration_skipped = False
    scc_summary: list[dict[str, Any]] = []

    if cycle_limit > 0:
        for i, c in enumerate(cyc_iter, 1):
            cycles.append(c)
            if i >= cycle_limit:
                break
    else:
        # Guard against pathological blow-ups: if graph is too large, skip enumeration and use SCCs
        EDGE_BUDGET = int(os.getenv("GRAPH_EDGE_BUDGET", "50000"))
        if gp.number_of_edges() > EDGE_BUDGET:
            cycle_enumeration_skipped = True
            sccs = [list(s) for s in nx.strongly_connected_components(gp) if len(s) > 1]
            scc_summary = [{"members": s, "size": len(s)} for s in sccs]
        else:
            cycles = [c for c in cyc_iter]

    if cycle_enumeration_skipped:
        centrality = nx.degree_centrality(gp) if gp.number_of_nodes() else {}
        violations: list[list[str]] = []
    else:
        centrality = nx.degree_centrality(g) if g.number_of_nodes() else {}
        # layer policy violations (dependencies should point inward in the layer order)
        order = layers.get("order", [])
        pkg2layer = cast(dict[str, str], layers.get("packages", {}) or {})
        rank = {name: i for i, name in enumerate(order)}
        violations = []
        rules = layers.get("rules", {})
        allow_outward = bool(rules.get("allow_outward", False))
        for u, v in g.edges():
            lu, lv = pkg2layer.get(u), pkg2layer.get(v)
            if lu and lv and lu in rank and lv in rank:
                if rank[lv] > rank[lu] and not allow_outward:
                    violations.append([lu, lv, f"edge:{u}->{v}"])

    result: dict[str, Any] = {
        "cycles": cycles,
        "centrality": centrality,
        "layer_violations": violations,
        "cycle_enumeration_skipped": cycle_enumeration_skipped,
    }
    if scc_summary:
        result["scc_summary"] = scc_summary
    return result


def style_and_render(
    g: nx.DiGraph, layers: dict[str, Any], analysis: dict[str, Any], out_svg: Path, fmt: str = "svg"
) -> None:
    """Compute style and render.

    Carry out the style and render operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    g : nx.DiGraph
        Description for ``g``.
    layers : collections.abc.Mapping
        Description for ``layers``.
    analysis : collections.abc.Mapping
        Description for ``analysis``.
    out_svg : Path
        Description for ``out_svg``.
    fmt : str | None
        Description for ``fmt``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import style_and_render
    >>> style_and_render(..., ..., ..., ...)  # doctest: +ELLIPSIS
    """
    
    pkg2layer = layers.get("packages", {}) or {}
    palette = {
        "domain": "#2f855a",
        "application": "#3182ce",
        "interface": "#805ad5",
        "infra": "#dd6b20",
    }

    cent = analysis["centrality"] or {}
    values = list(cent.values())
    q80 = sorted(values)[int(0.80 * len(values))] if values else 0.0

    cycle_edges = set()
    for cyc in analysis["cycles"]:
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            cycle_edges.add((u, v))

    pd = pydot.Dot(graph_type="digraph", rankdir="LR")

    # nodes
    for n in sorted(g.nodes()):
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
        edge_color = "#e53e3e" if (u, v) in cycle_edges else "#a0aec0"
        penw = "2.5" if (u, v) in cycle_edges else "1.2"
        label = f"{layer_u}→{layer_v}" if layer_u != layer_v else ""
        pd.add_edge(pydot.Edge(u, v, color=edge_color, penwidth=penw, fontsize="8", label=label))

    data = pd.create_svg() if fmt == "svg" else pd.create_png()
    out_svg.write_bytes(data)


def write_meta(meta: dict[str, Any], out_json: Path) -> None:
    """Compute write meta.

    Carry out the write meta operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    meta : collections.abc.Mapping
        Description for ``meta``.
    out_json : Path
        Description for ``out_json``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import write_meta
    >>> write_meta(..., ...)  # doctest: +ELLIPSIS
    """
    
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def enforce_policy(
    analysis: dict[str, Any], allow: dict[str, Any], fail_cycles: bool, fail_layers: bool
) -> None:
    """Compute enforce policy.

    Carry out the enforce policy operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    analysis : collections.abc.Mapping
        Description for ``analysis``.
    allow : collections.abc.Mapping
        Description for ``allow``.
    fail_cycles : bool
        Description for ``fail_cycles``.
    fail_layers : bool
        Description for ``fail_layers``.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import enforce_policy
    >>> enforce_policy(..., ..., ..., ...)  # doctest: +ELLIPSIS
    """
    
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


# --------------------------------------------------------------------------------------
# NEW: cache helpers + parallel worker
# --------------------------------------------------------------------------------------


def package_snapshot_digest(pkg: str) -> str:
    """Compute package snapshot digest.

    Carry out the package snapshot digest operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    pkg : str
        Description for ``pkg``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.docs.build_graphs import package_snapshot_digest
    >>> result = package_snapshot_digest(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """

    h = hashlib.sha256()
    path = SRC / pkg
    if not path.exists():
        return "EMPTY"
    for entry in sorted(path.rglob("*")):
        if not entry.is_file():
            continue
        st = entry.stat()
        h.update(str(entry.relative_to(ROOT)).encode())
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest() or "EMPTY"


def last_tree_commit(pkg: str) -> str:
    """Compute last tree commit.

    Carry out the last tree commit operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.

    Parameters
    ----------
    pkg : str
        Description for ``pkg``.

    Returns
    -------
    str
        Description of return value.

    Examples
    --------
    >>> from tools.docs.build_graphs import last_tree_commit
    >>> result = last_tree_commit(...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """

    sha = ""
    try:
        sha = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", f"src/{pkg}"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        sha = ""

    dirty = False
    if sha:
        try:
            dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--", f"src/{pkg}"],
                    cwd=str(ROOT),
                    text=True,
                ).strip()
            )
        except Exception:
            dirty = True
    else:
        dirty = True

    snapshot_digest = package_snapshot_digest(pkg)
    if not sha:
        return snapshot_digest
    if not dirty:
        return sha

    blended = hashlib.sha256()
    blended.update(sha.encode())
    blended.update(snapshot_digest.encode())
    return blended.hexdigest()


def cache_bucket(cache_dir: Path, pkg: str, tree_hash: str) -> Path:
    """Compute cache bucket.

    Carry out the cache bucket operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    cache_dir : Path
        Description for ``cache_dir``.
    pkg : str
        Description for ``pkg``.
    tree_hash : str
        Description for ``tree_hash``.
    
    Returns
    -------
    Path
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import cache_bucket
    >>> result = cache_bucket(..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    return cache_dir / pkg / tree_hash


def build_one_package(
    pkg: str,
    fmt: str,
    excludes: list[str],
    max_bacon: int,
    cache_dir: Path,
    use_cache: bool,
    verbose: bool,
) -> tuple[str, bool, bool, bool]:
    """Compute build one package.

    Carry out the build one package operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Parameters
    ----------
    pkg : str
        Description for ``pkg``.
    fmt : str
        Description for ``fmt``.
    excludes : List[str]
        Description for ``excludes``.
    max_bacon : int
        Description for ``max_bacon``.
    cache_dir : Path
        Description for ``cache_dir``.
    use_cache : bool
        Description for ``use_cache``.
    verbose : bool
        Description for ``verbose``.
    
    Returns
    -------
    Tuple[str, bool, bool, bool]
        Description of return value.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import build_one_package
    >>> result = build_one_package(..., ..., ..., ..., ..., ..., ...)
    >>> result  # doctest: +ELLIPSIS
    ...
    """
    
    used_cache = False
    pydeps_ok = True
    pyrev_ok = True

    imports_out = OUT / f"{pkg}-imports.{fmt}"
    uml_out = OUT / f"{pkg}-uml.{fmt}"

    tree_h: str | None = None
    bucket: Path | None = None
    if use_cache:
        tree_h = last_tree_commit(pkg)
        bucket = cache_bucket(cache_dir, pkg, tree_h)
        cache_imp = bucket / imports_out.name
        cache_uml = bucket / uml_out.name
        if cache_imp.exists() and cache_uml.exists():
            shutil.copy2(cache_imp, imports_out)
            shutil.copy2(cache_uml, uml_out)
            used_cache = True
            if verbose:
                print(f"[graphs] cache hit: {pkg}@{tree_h[:7]}")
            return (pkg, True, pydeps_ok, pyrev_ok)

    # Build fresh
    imports_out.unlink(missing_ok=True)
    uml_out.unlink(missing_ok=True)
    try:
        build_pydeps_for_package(pkg, imports_out, excludes, max_bacon, fmt)
    except Exception:
        pydeps_ok = False
    try:
        build_pyreverse_for_package(pkg, OUT, fmt)
    except Exception:
        pyrev_ok = False

    if not (pydeps_ok and pyrev_ok):
        for partial in (imports_out, uml_out):
            if partial.exists():
                partial.unlink()

    # Save to cache if requested and successful
    if use_cache and pydeps_ok and pyrev_ok and tree_h and bucket is not None:
        bucket.mkdir(parents=True, exist_ok=True)
        shutil.copy2(imports_out, bucket / imports_out.name)
        shutil.copy2(uml_out, bucket / uml_out.name)
        if verbose:
            print(f"[graphs] cached: {pkg}@{tree_h[:7]}")
    elif not (pydeps_ok and pyrev_ok):
        imports_out.unlink(missing_ok=True)
        uml_out.unlink(missing_ok=True)

    return (pkg, used_cache, pydeps_ok, pyrev_ok)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    """Compute main.

    Carry out the main operation for the surrounding component. Generated documentation highlights how this helper collaborates with neighbouring utilities. Callers rely on the routine to remain stable across releases.
    
    Examples
    --------
    >>> from tools.docs.build_graphs import main
    >>> main()  # doctest: +ELLIPSIS
    """
    
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

    ensure_bin("pydeps")
    ensure_bin("pyreverse")

    packages: list[str] = (
        [s.strip() for s in args.packages.split(",") if s.strip()]
        if args.packages
        else find_top_packages()
    )
    excludes = args.exclude or ["tests/.*", "site/.*"]
    fmt = args.format
    use_cache = not args.no_cache
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"[graphs] packages={packages}")
        print(f"[graphs] cache_dir={cache_dir} use_cache={use_cache}")

    # 1) Per-package graphs (PARALLEL + CACHE)
    t0 = time.time()
    max_workers = args.max_workers or os.cpu_count() or 1
    results: list[tuple[str, bool, bool, bool]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                build_one_package,
                pkg,
                fmt,
                excludes,
                args.max_bacon,
                cache_dir,
                use_cache,
                args.verbose,
            )
            for pkg in packages
        ]
        for fut in as_completed(futs):
            pkg, used_cache, ok1, ok2 = fut.result()
            results.append((pkg, used_cache, ok1, ok2))
            if args.verbose:
                src = "cache" if used_cache else "built"
                print(
                    f"[graphs] {pkg}: {src}; pydeps={'ok' if ok1 else 'FAIL'}; pyreverse={'ok' if ok2 else 'FAIL'}"
                )

    failures: list[str] = []
    for pkg, _, ok1, ok2 in results:
        reasons: list[str] = []
        if not ok1:
            reasons.append("pydeps")
        if not ok2:
            reasons.append("pyreverse")
        if reasons:
            failures.append(f"{pkg} ({', '.join(reasons)})")

    if failures:
        print(
            f"[graphs] per-package build failures: {', '.join(sorted(failures))}",
            file=sys.stderr,
        )
    failures: list[tuple[str, bool, bool]] = [
        (pkg, ok1, ok2)
        for pkg, _, ok1, ok2 in results
        if not (ok1 and ok2)
    ]
    if failures:
        lines = ["[graphs] build failures detected during per-package graph generation:"]
        for pkg, ok1, ok2 in failures:
            failed_parts: list[str] = []
            if not ok1:
                failed_parts.append("pydeps")
            if not ok2:
                failed_parts.append("pyreverse")
            parts = ", ".join(failed_parts)
            lines.append(f" - {pkg}: {parts} failed")
        print("\n".join(lines), file=sys.stderr)
        sys.exit(3)

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
    style_and_render(g, layers, analysis, OUT / f"subsystems.{fmt}", fmt=fmt)
    meta: dict[str, Any] = {
        "packages": sorted([str(n) for n in g.nodes()]),
        "cycles": analysis["cycles"],
        "centrality": analysis["centrality"],
        "layer_violations": analysis["layer_violations"],
        "cycle_enumeration_skipped": analysis.get("cycle_enumeration_skipped", False),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if analysis.get("scc_summary"):
        meta["scc_summary"] = analysis["scc_summary"]
    (OUT / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary_lines = [
        "# Subsystem Graph Metadata",
        "",
        (
            "*Cycle enumeration skipped:* Yes"
            if meta["cycle_enumeration_skipped"]
            else "*Cycle enumeration skipped:* No"
        ),
    ]
    if meta.get("scc_summary"):
        summary_lines += [
            "",
            "The fallback strongly connected components are:",
        ]
        for entry in meta["scc_summary"]:
            members = ", ".join(sorted(entry["members"]))
            summary_lines.append(f"- size {entry['size']}: {members}")
    else:
        summary_lines.append("")
        summary_lines.append("Cycle enumeration completed normally.")
    (OUT / "subsystems_meta.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # 5) Enforce policy
    enforce_policy(analysis, allow, args.fail_on_cycles, args.fail_on_layer_violations)

    if args.verbose:
        dt = time.time() - t0
        built = sum(1 for _, c, _, _ in results if not c)
        cached = sum(1 for _, c, _, _ in results if c)
        print(
            f"[graphs] done in {dt:.2f}s; per-package built={built} cached={cached}; outputs in {OUT}"
        )


if __name__ == "__main__":
    main()
