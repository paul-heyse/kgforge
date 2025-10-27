from __future__ import annotations

from multiprocessing import Process

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tests.mock_servers.openalex_mock import app as openalex_app
from tests.mock_servers.unpaywall_mock import app as unpaywall_app


def run_openalex():
    uvicorn.run(openalex_app, host="0.0.0.0", port=8998, log_level="warning")


def run_unpaywall():
    uvicorn.run(unpaywall_app, host="0.0.0.0", port=8997, log_level="warning")


def run_pdf_files():
    app = FastAPI()
    app.mount("/pdf", StaticFiles(directory="tests/fixtures/pdf"), name="pdf")
    uvicorn.run(app, host="0.0.0.0", port=8999, log_level="warning")


if __name__ == "__main__":
    procs = [
        Process(target=run_openalex),
        Process(target=run_unpaywall),
        Process(target=run_pdf_files),
    ]
    [p.start() for p in procs]
    [p.join() for p in procs]
