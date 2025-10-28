from pathlib import Path

from kgfoundry.registry.helper import DuckDBRegistryHelper
from kgfoundry.registry.migrate import apply


def test_registry_two_phase(tmp_path: Path) -> None:
    db = tmp_path / "catalog.duckdb"
    # apply migration
    apply(str(db), "registry/migrations")
    reg = DuckDBRegistryHelper(str(db))
    run_id = reg.new_run("dense_embed", "Qwen3-Embedding-4B", "main", {"dim": 2560})
    ds = reg.begin_dataset("dense", run_id)
    reg.commit_dataset(ds, "/data/parquet/dense/model=Qwen3-Embedding-4B/run_id=...", rows=1)
    reg.close_run(run_id, True)
