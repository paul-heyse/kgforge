# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [API](#api)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.cli`** — Module for orchestration.cli → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:1:1) | [view](cli.py#L1)
  - **`orchestration.cli.api`** — Run the FastAPI app → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:90:1) | [view](cli.py#L90-L95)
  - **`orchestration.cli.index_bm25`** — Build a BM25 index from chunk fixtures (id, title, section, body) → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:16:1) | [view](cli.py#L16-L58)
  - **`orchestration.cli.index_faiss`** — Train & build FAISS index from fixture dense vectors → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:61:1) | [view](cli.py#L61-L87)
- **`orchestration.fixture_flow`** — Module for orchestration.fixture_flow → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:1:1) | [view](fixture_flow.py#L1)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Fixture pipeline → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:148:1) | [view](fixture_flow.py#L148-L163)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — T prepare dirs → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:13:1) | [view](fixture_flow.py#L13-L28)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — T register in duckdb → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:97:1) | [view](fixture_flow.py#L97-L145)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — T write fixture chunks → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:31:1) | [view](fixture_flow.py#L31-L56)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — T write fixture dense → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:59:1) | [view](fixture_flow.py#L59-L74)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — T write fixture splade → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:77:1) | [view](fixture_flow.py#L77-L94)
- **`orchestration.flows`** — Module for orchestration.flows → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:1:1) | [view](flows.py#L1)
  - **`orchestration.flows.e2e_flow`** — E2e flow → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:21:1) | [view](flows.py#L21-L39)
  - **`orchestration.flows.t_echo`** — T echo → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:8:1) | [view](flows.py#L8-L18)
