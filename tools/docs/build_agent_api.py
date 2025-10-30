"""Emit OpenAPI documentation describing the agent catalog callable API."""

from __future__ import annotations

import json
from pathlib import Path

OUTPUT = Path("docs/_build/agent_api_openapi.json")


def build_spec() -> dict[str, object]:
    """Return the OpenAPI specification document."""
    problem_details = {
        "type": "object",
        "description": "RFC 9457 Problem Details error response.",
        "properties": {
            "type": {"type": "string", "format": "uri"},
            "title": {"type": "string"},
            "status": {"type": "integer"},
            "detail": {"type": "string"},
            "instance": {"type": "string", "format": "uri", "nullable": True},
        },
        "required": ["type", "title", "status"],
    }
    search_result = {
        "type": "object",
        "properties": {
            "symbol_id": {"type": "string"},
            "score": {"type": "number"},
            "lexical_score": {"type": "number"},
            "vector_score": {"type": "number"},
            "package": {"type": "string"},
            "module": {"type": "string"},
            "qname": {"type": "string"},
        },
        "required": ["symbol_id", "score", "package", "module", "qname"],
    }
    symbol_response = {
        "type": "object",
        "description": "Catalog symbol metadata.",
        "properties": {
            "qname": {"type": "string"},
            "kind": {"type": "string"},
            "symbol_id": {"type": "string"},
            "anchors": {
                "type": "object",
                "properties": {
                    "start_line": {"type": "integer", "nullable": True},
                    "end_line": {"type": "integer", "nullable": True},
                    "cst_fingerprint": {"type": "string", "nullable": True},
                    "remap_order": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
            "quality": {"type": "object"},
            "metrics": {"type": "object"},
            "change_impact": {"type": "object"},
            "agent_hints": {"type": "object"},
        },
    }
    return {
        "openapi": "3.2.0",
        "info": {
            "title": "kgfoundry Agent Catalog API",
            "version": "1.0.0",
            "description": "Callable operations for the catalogctl stdio server.",
        },
        "paths": {
            "/catalog/search": {
                "post": {
                    "operationId": "catalog_search",
                    "summary": "Hybrid lexical search",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "k": {"type": "integer", "default": 10},
                                        "facets": {
                                            "type": "object",
                                            "additionalProperties": {"type": "string"},
                                        },
                                    },
                                    "required": ["query"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "array", "items": search_result}
                                }
                            },
                        },
                        "400": {
                            "description": "Invalid request",
                            "content": {"application/json": {"schema": problem_details}},
                        },
                    },
                }
            },
            "/catalog/symbol": {
                "get": {
                    "operationId": "catalog_symbol",
                    "summary": "Fetch symbol metadata",
                    "parameters": [
                        {
                            "in": "query",
                            "name": "symbol_id",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Symbol payload",
                            "content": {"application/json": {"schema": symbol_response}},
                        },
                        "404": {
                            "description": "Symbol not found",
                            "content": {"application/json": {"schema": problem_details}},
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "ProblemDetails": problem_details,
                "SearchResult": search_result,
                "Symbol": symbol_response,
            }
        },
    }


def main() -> int:
    """Write the OpenAPI document to the repository build directory."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec()
    OUTPUT.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
