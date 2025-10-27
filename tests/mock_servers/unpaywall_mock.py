from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Unpaywall Mock")


@app.get("/v2/{doi}")
def by_doi(doi: str, email: str = "test@example.com"):
    return {
        "doi": doi,
        "best_oa_location": {
            "url_for_pdf": f"http://localhost:8999/pdf/{doi.replace('/', '_')}.pdf"
        },
    }
