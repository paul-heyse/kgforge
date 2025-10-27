from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="OpenAlex Mock")


@app.get("/works")
def works(topic: str = "test", per_page: int = 2, cursor: str = "*"):
    return {
        "meta": {"count": 2, "per_page": per_page, "next_cursor": None},
        "results": [
            {
                "id": "W1",
                "doi": "10.1000/test1",
                "title": "Test work 1 about " + topic,
                "best_oa_location": {
                    "pdf_url": "http://localhost:8999/pdf/test1.pdf",
                    "is_oa": True,
                    "version": "publishedVersion",
                },
                "locations": [],
            },
            {
                "id": "W2",
                "doi": "10.1000/test2",
                "title": "Test work 2 about " + topic,
                "best_oa_location": None,
                "locations": [
                    {
                        "pdf_url": "http://localhost:8999/pdf/test2.pdf",
                        "is_oa": True,
                        "version": "acceptedVersion",
                    }
                ],
            },
        ],
    }
