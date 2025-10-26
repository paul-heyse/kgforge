
from __future__ import annotations
import argparse, duckdb, pathlib

def apply(db: str, migrations_dir: str) -> None:
    con = duckdb.connect(db)
    for p in sorted(pathlib.Path(migrations_dir).glob("*.sql")):
        con.execute(p.read_text())
    con.close()

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("apply")
    a.add_argument("--db", required=True)
    a.add_argument("--migrations", required=True)
    ns = ap.parse_args()
    if ns.cmd == "apply":
        apply(ns.db, ns.migrations)

if __name__ == "__main__":
    main()
