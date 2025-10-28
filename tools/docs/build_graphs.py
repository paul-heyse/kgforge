"""Build Graphs utilities."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "docs" / "_build" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)
GRAPH_PACKAGES = {
    "kgfoundry",
    "kgfoundry_common",
    "kg_builder",
    "search_api",
    "embeddings_dense",
    "embeddings_sparse",
    "orchestration",
    "registry",
}


def run(cmd: list[str]) -> None:
    """Compute run.

    Carry out the run operation.

    Parameters
    ----------
    cmd : List[str]
        Description for ``cmd``.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    




















    subprocess.run(cmd, check=True, cwd=ROOT)


def top_level_packages() -> set[str]:
    """Compute top level packages.

    Carry out the top level packages operation.

    Returns
    -------
    Set[str]
        Description of return value.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    




















    packages: set[str] = set()
    if not SRC.exists():
        return packages
    for init_file in SRC.rglob("__init__.py"):
        rel = init_file.relative_to(SRC)
        if not rel.parts:
            continue
        top = rel.parts[0]
        if top == "__init__.py":
            continue
        if GRAPH_PACKAGES and top not in GRAPH_PACKAGES:
            continue
        packages.add(top)
    return packages


def build_pydeps(pkg: str) -> None:
    """Compute build pydeps.

    Carry out the build pydeps operation.

    Parameters
    ----------
    pkg : str
        Description for ``pkg``.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    




















    dot_path = OUT / f"{pkg}.dot"
    svg_path = OUT / f"{pkg}-imports.svg"
    cmd = [
        sys.executable,
        "-m",
        "pydeps",
        f"src/{pkg}",
        "--max-bacon=4",
        "--noshow",
        "-T",
        "dot",
        "-o",
        str(dot_path),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True, cwd=ROOT)
    finally:
        dot_path.unlink(missing_ok=True)


def build_pyreverse(pkg: str) -> None:
    """Compute build pyreverse.

    Carry out the build pyreverse operation.

    Parameters
    ----------
    pkg : str
        Description for ``pkg``.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    




















    cmd = [
        sys.executable,
        "-m",
        "pylint.pyreverse.main",
        f"src/{pkg}",
        "-o",
        "svg",
        "-d",
        str(OUT),
        "-p",
        pkg,
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    for svg in OUT.glob("classes_*.svg"):
        target = OUT / f"{pkg}-uml.svg"
        svg.replace(target)
    for svg in OUT.glob("packages_*.svg"):
        svg.unlink()


def main() -> None:
    """Compute main.

    Carry out the main operation.
    """
    
    
    
    
    
    
    
    
    
    
    
    
    




















    for pkg in sorted(top_level_packages()):
        try:
            build_pydeps(pkg)
            build_pyreverse(pkg)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - diagnostic aid
            print(f"[build_graphs] Skipping {pkg} due to error: {exc}")


if __name__ == "__main__":
    main()
