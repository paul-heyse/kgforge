
from __future__ import annotations
from typing import List, Optional, Dict
import uuid, json
import duckdb
from kgforge.kgforge_common.models import Doc, DoctagsAsset, Chunk, LinkAssertion

class DuckDBRegistryHelper:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _con(self):
        return duckdb.connect(self.db_path)

    def new_run(self, purpose: str, model_id: Optional[str], revision: Optional[str], config: dict) -> str:
        run_id = str(uuid.uuid4())
        con = self._con()
        con.execute(
            "INSERT INTO runs (run_id,purpose,model_id,revision,started_at,config) VALUES (?,?,?,?,CURRENT_TIMESTAMP,?)",
            [run_id, purpose, model_id, revision, json.dumps(config)],
        )
        con.close()
        return run_id

    def close_run(self, run_id: str, success: bool, notes: Optional[str] = None) -> None:
        con = self._con()
        con.execute("UPDATE runs SET finished_at=CURRENT_TIMESTAMP WHERE run_id=?", [run_id])
        con.execute("INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [str(uuid.uuid4()), "RunClosed", run_id, json.dumps({"success": success, "notes": notes or ""})])
        con.close()

    def begin_dataset(self, kind: str, run_id: str) -> str:
        dataset_id = str(uuid.uuid4())
        con = self._con()
        con.execute("INSERT INTO datasets (dataset_id,kind,parquet_root,run_id,created_at) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [dataset_id, kind, "", run_id])
        con.close()
        return dataset_id

    def commit_dataset(self, dataset_id: str, parquet_root: str, rows: int) -> None:
        con = self._con()
        con.execute("UPDATE datasets SET parquet_root=? WHERE dataset_id=?", [parquet_root, dataset_id])
        con.execute("INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [str(uuid.uuid4()), "DatasetCommitted", dataset_id, json.dumps({"rows": rows, "root": parquet_root})])
        con.close()

    def rollback_dataset(self, dataset_id: str) -> None:
        con = self._con()
        con.execute("DELETE FROM datasets WHERE dataset_id=?", [dataset_id])
        con.execute("INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [str(uuid.uuid4()), "DatasetRolledBack", dataset_id, "{}"])
        con.close()

    def register_documents(self, docs: List[Doc]) -> None:
        con = self._con()
        for d in docs:
            con.execute(
                """INSERT OR REPLACE INTO documents
                (doc_id, openalex_id, doi, arxiv_id, pmcid, title, authors, pub_date, license,
                 language, pdf_uri, source, content_hash, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)""",
                [d.id, d.openalex_id, d.doi, d.arxiv_id, d.pmcid, d.title, json.dumps(d.authors),
                 d.pub_date, d.license, d.language, d.pdf_uri, d.source, d.content_hash or ""],
            )
        con.close()

    def register_doctags(self, assets: List[DoctagsAsset]) -> None:
        con = self._con()
        for a in assets:
            con.execute("INSERT OR REPLACE INTO doctags VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)",
                        [a.doc_id, a.doctags_uri, a.pages, a.vlm_model, a.vlm_revision, a.avg_logprob])
        con.close()

    def emit_event(self, event_name: str, subject_id: str, payload: Dict) -> None:
        con = self._con()
        con.execute("INSERT INTO pipeline_events VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [str(uuid.uuid4()), event_name, subject_id, json.dumps(payload)])
        con.close()
