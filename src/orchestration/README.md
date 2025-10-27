# `orchestration`

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## API
- **`orchestration.cli`** — Module for orchestration.cli → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:1:1) | [view](cli.py#L1)
  - **`orchestration.cli.api`** — Run the FastAPI app → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:96:1) | [view](cli.py#L96-L101)
  - **`orchestration.cli.index_bm25`** — Build a BM25 index from chunk fixtures (id, title, section, body) → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:22:1) | [view](cli.py#L22-L64)
  - **`orchestration.cli.index_faiss`** — Train & build FAISS index from fixture dense vectors → [open](vscode://file//home/paul/KGForge/src/orchestration/cli.py:67:1) | [view](cli.py#L67-L93)
- **`orchestration.fixture_flow`** — Module for orchestration.fixture_flow → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:1:1) | [view](fixture_flow.py#L1)
  - **`orchestration.fixture_flow.fixture_pipeline`** — Run the end-to-end fixture pipeline → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:177:1) | [view](fixture_flow.py#L177-L194)
  - **`orchestration.fixture_flow.t_prepare_dirs`** — T prepare dirs → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:22:1) | [view](fixture_flow.py#L22-L41)
  - **`orchestration.fixture_flow.t_register_in_duckdb`** — T register in duckdb → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:122:1) | [view](fixture_flow.py#L122-L174)
  - **`orchestration.fixture_flow.t_write_fixture_chunks`** — T write fixture chunks → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:44:1) | [view](fixture_flow.py#L44-L73)
  - **`orchestration.fixture_flow.t_write_fixture_dense`** — T write fixture dense → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:76:1) | [view](fixture_flow.py#L76-L95)
  - **`orchestration.fixture_flow.t_write_fixture_splade`** — T write fixture splade → [open](vscode://file//home/paul/KGForge/src/orchestration/fixture_flow.py:98:1) | [view](fixture_flow.py#L98-L119)
- **`orchestration.flows`** — Module for orchestration.flows → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:1:1) | [view](flows.py#L1)
  - **`orchestration.flows.e2e_flow`** — E2e flow → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:30:1) | [view](flows.py#L30-L48)
  - **`orchestration.flows.t_echo`** — T echo → [open](vscode://file//home/paul/KGForge/src/orchestration/flows.py:13:1) | [view](flows.py#L13-L27)
