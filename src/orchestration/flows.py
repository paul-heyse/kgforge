
from __future__ import annotations
from prefect import flow, task

@task
def t_echo(msg: str) -> str:
    return msg

@flow(name="kgforge_e2e_skeleton")
def e2e_flow():
    return [t_echo.submit(x).result() for x in ["harvest","doctags","chunk","embed_dense","encode_splade","bm25","faiss","ontology","concept_embed","linker","kg"]]
