"""
Create a .importlinter file with a default layered-architecture contract.
Detects the root package automatically (prefers src/<pkg>).
"""
from pathlib import Path

from detect_pkg import detect_primary

ROOT = Path(__file__).resolve().parents[1]

pkg = detect_primary()
tmpl = f"""[importlinter]
root_package = {pkg}

[importlinter:contract:layers]
name = Respect layered architecture
type = layers
layers =
    {pkg}.presentation
    {pkg}.domain
    {pkg}.infrastructure
"""
out = ROOT / ".importlinter"
out.write_text(tmpl, encoding="utf-8")
print(f"Wrote {out}")
