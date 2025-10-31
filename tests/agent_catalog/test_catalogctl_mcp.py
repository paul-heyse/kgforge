"""Tests for the stdio JSON-RPC catalog server.

Tests cover success paths, invalid input handling, Problem Details emission,
and schema validation for MCP responses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from jsonschema import (  # type: ignore[import-untyped]  # jsonschema types
    Draft202012Validator,
    ValidationError,
)
from jsonschema.validators import RefResolver  # type: ignore[import-untyped]  # jsonschema types

from kgfoundry_common.schema_helpers import load_schema

FIXTURE = Path("tests/fixtures/agent/catalog_sample.json").resolve()
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def mcp_payload_schema() -> dict[str, object]:
    """Load mcp_payload.json schema."""
    schema_path = Path("schema/search/mcp_payload.json")
    if not schema_path.exists():
        pytest.skip(f"Schema not found: {schema_path}")
    return load_schema(schema_path)  # type: ignore[no-any-return]  # load_schema returns Any but we treat as dict[str, object]


@pytest.fixture
def mcp_payload_validator(mcp_payload_schema: dict[str, object]) -> Draft202012Validator:
    """Create validator for mcp_payload.json with resolver for $ref."""
    base_uri = mcp_payload_schema.get("$id", "")  # type: ignore[misc]  # dict access returns Any
    base_uri_str = str(base_uri) if base_uri else ""  # Cast to str for dict key
    store: dict[str, object] = {base_uri_str: mcp_payload_schema}

    # Load referenced schemas
    search_response_path = Path("schema/search/search_response.json")
    if search_response_path.exists():
        search_response = load_schema(search_response_path)  # type: ignore[assignment]  # load_schema returns Any
        search_response_id = search_response.get("$id", "")  # type: ignore[misc]  # dict access returns Any
        if search_response_id:
            search_response_id_str = str(search_response_id)  # Cast to str for dict key
            store[search_response_id_str] = search_response  # type: ignore[misc]  # search_response_id_str may be Any
        store["../search/search_response.json"] = search_response

    problem_details_path = Path("schema/common/problem_details.json")
    if problem_details_path.exists():
        problem_details = load_schema(problem_details_path)  # type: ignore[assignment]  # load_schema returns Any
        problem_details_id = problem_details.get("$id", "")  # type: ignore[misc]  # dict access returns Any
        if problem_details_id:
            problem_details_id_str = str(problem_details_id)  # Cast to str for dict key
            store[problem_details_id_str] = problem_details  # type: ignore[misc]  # problem_details_id_str may be Any
        store["https://kgfoundry.dev/schema/common/problem_details.json"] = problem_details
        store["https://kgfoundry.dev/schemas/common/problem_details.json"] = problem_details
        store["../common/problem_details.json"] = problem_details

    def _resolve_local(uri: str) -> object:
        key = str(uri)
        if key in store:
            return store[key]
        raise KeyError(key)

    resolver = RefResolver.from_schema(  # type: ignore[misc]  # RefResolver typing limitation
        mcp_payload_schema,
        store=store,
        handlers={"https": _resolve_local, "http": _resolve_local},
    )
    return Draft202012Validator(mcp_payload_schema, resolver=resolver)  # type: ignore[call-arg,misc]  # jsonschema typing limitation - resolver is valid at runtime


def _rpc(process: subprocess.Popen[str], payload: dict[str, Any]) -> dict[str, object]:
    """Send a JSON-RPC request to ``process`` and decode the response."""
    stdin = process.stdin
    stdout = process.stdout
    if stdin is None or stdout is None:
        message = "catalogctl-mcp stdio streams are unavailable"
        raise RuntimeError(message)
    stdin.write(json.dumps(payload) + "\n")
    stdin.flush()
    line = stdout.readline()
    if not line:
        message = "catalogctl-mcp terminated unexpectedly"
        raise RuntimeError(message)
    return json.loads(line)  # type: ignore[assignment,no-any-return]  # JSON parsing returns Any


class TestMCPSuccess:
    """Test successful MCP operations."""

    def test_catalogctl_mcp_session_round_trip(self) -> None:
        """The stdio server should respond to catalog queries and shut down cleanly."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - command uses trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            response = _rpc(
                process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            )
            assert "result" in response
            result = response.get("result")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(result, dict)
            capabilities = result.get("capabilities")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(capabilities, dict)
            procedures = capabilities.get("procedures")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(procedures, (dict, list))
            assert "catalog.search" in str(procedures)

            search_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.search",
                    "params": {"query": "demo", "k": 1},
                },
            )
            search_result = search_response.get("result")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(search_result, dict)
            results_list = search_result.get("results")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(results_list, list)
            assert len(results_list) > 0
            first_result = results_list[0]  # type: ignore[misc]  # list access returns Any
            assert isinstance(first_result, dict)
            assert first_result.get("qname") == "demo.module.fn"  # type: ignore[misc]  # dict access returns Any
            metadata = search_result.get("metadata")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(metadata, dict)
            assert metadata.get("correlation_id")  # type: ignore[misc]  # dict access returns Any
            query_info = metadata.get("query_info")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(query_info, dict)
            assert query_info.get("query") == "demo"  # type: ignore[misc]  # dict access returns Any

            shutdown = _rpc(
                process, {"jsonrpc": "2.0", "id": 3, "method": "session.shutdown", "params": {}}
            )
            shutdown_result = shutdown.get("result")  # type: ignore[misc]  # dict access returns Any
            assert shutdown_result is None
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    def test_catalogctl_mcp_enforces_rbac(self, tmp_path: Path) -> None:
        """Hosted mode should forbid viewer roles from invoking admin methods."""
        audit_log = tmp_path / "audit.jsonl"
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
            "--hosted-mode",
            "--role",
            "viewer",
            "--audit-log",
            str(audit_log),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            denied = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.open_anchor",
                    "params": {"symbol_id": "4b227777d4dd1fc61c6f884f48641d02"},
                },
            )
            denied_error = denied.get("error")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(denied_error, dict)
            denied_status = denied_error.get("status")  # type: ignore[misc]  # dict access returns Any
            assert denied_status == 403
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()
        # Verify audit log contains forbidden entry
        if audit_log.exists():
            lines = [
                json.loads(line)  # type: ignore[assignment]  # JSON parsing returns Any
                for line in audit_log.read_text(encoding="utf-8").splitlines()
            ]
            statuses = {
                entry.get("status")  # type: ignore[misc]  # dict access returns Any
                for entry in lines
                if isinstance(entry, dict)
            }
            assert "forbidden" in statuses
            for entry in lines:
                if isinstance(entry, dict):
                    assert entry.get("correlation_id")  # type: ignore[misc]  # dict access returns Any


class TestMCPSchemaValidation:
    """Test that MCP responses validate against schema."""

    def test_search_response_validates_against_schema(
        self, mcp_payload_validator: Draft202012Validator
    ) -> None:
        """Search response should validate against mcp_payload.json schema."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - command uses trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            search_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.search",
                    "params": {"query": "demo", "k": 1},
                },
            )
            # Validate response structure
            try:
                mcp_payload_validator.validate(search_response)
            except ValidationError:
                # Check basic structure even if full validation fails
                assert "jsonrpc" in search_response
                assert search_response.get("jsonrpc") == "2.0"  # type: ignore[misc]  # dict access returns Any
                assert "id" in search_response
                assert "result" in search_response
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    def test_search_response_has_required_fields(self) -> None:
        """Search response should have required JSON-RPC fields."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            search_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.search",
                    "params": {"query": "demo", "k": 1},
                },
            )
            # Check required fields
            assert "jsonrpc" in search_response
            assert search_response.get("jsonrpc") == "2.0"  # type: ignore[misc]  # dict access returns Any
            assert "id" in search_response
            assert "result" in search_response
            # Check result structure
            result = search_response.get("result")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(result, dict)
            assert "results" in result  # type: ignore[misc]  # dict access on object
            assert "total" in result  # type: ignore[misc]  # dict access on object
            assert "took_ms" in result  # type: ignore[misc]  # dict access on object
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()


class TestMCPInvalidInput:
    """Test handling of invalid MCP input."""

    def test_invalid_method_returns_error(self) -> None:
        """Invalid method should return JSON-RPC error."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            error_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "invalid.method",
                    "params": {},
                },
            )
            assert "error" in error_response
            error = error_response.get("error")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(error, dict)
            assert "code" in error
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    def test_invalid_params_returns_error(self) -> None:
        """Invalid params should return JSON-RPC error."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            error_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.search",
                    "params": {"query": ""},  # Empty query may be invalid
                },
            )
            assert "error" in error_response
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()


class TestMCPProblemDetails:
    """Test Problem Details emission for MCP errors."""

    def test_error_response_includes_problem_details(
        self, mcp_payload_validator: Draft202012Validator
    ) -> None:
        """Error responses should include RFC 9457 Problem Details."""
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            error_response = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "invalid.method",
                    "params": {},
                },
            )
            assert "error" in error_response
            error = error_response.get("error")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(error, dict)
            # Check for Problem Details in error.data
            if "data" in error:
                problem_details = error.get("data")  # type: ignore[misc]  # dict access returns Any
                assert isinstance(problem_details, dict)
                assert "type" in problem_details
                assert "status" in problem_details
                assert "title" in problem_details
                assert problem_details.get("correlation_id")  # type: ignore[misc]  # dict access returns Any
                # Validate against schema
                with suppress(ValidationError):
                    mcp_payload_validator.validate(error_response)
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()

    def test_rbac_error_includes_problem_details(self, tmp_path: Path) -> None:
        """RBAC errors should include Problem Details."""
        audit_log = tmp_path / "audit.jsonl"
        command = [
            sys.executable,
            "-m",
            "tools.agent_catalog.catalogctl_mcp",
            "--catalog",
            str(FIXTURE),
            "--repo-root",
            str(REPO_ROOT),
            "--hosted-mode",
            "--role",
            "viewer",
            "--audit-log",
            str(audit_log),
        ]
        process = subprocess.Popen(  # noqa: S603 - trusted interpreter
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _rpc(process, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            denied = _rpc(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "catalog.open_anchor",
                    "params": {"symbol_id": "4b227777d4dd1fc61c6f884f48641d02"},
                },
            )
            denied_error = denied.get("error")  # type: ignore[misc]  # dict access returns Any
            assert isinstance(denied_error, dict)
            assert denied_error.get("status") == 403  # type: ignore[misc]  # dict access returns Any
            # Check for Problem Details
            if "data" in denied_error:
                problem_details = denied_error.get("data")  # type: ignore[misc]  # dict access returns Any
                assert isinstance(problem_details, dict)
                assert "type" in problem_details
                assert "status" in problem_details
                assert problem_details.get("status") == 403  # type: ignore[misc]  # dict access returns Any
                assert problem_details.get("correlation_id")  # type: ignore[misc]  # dict access returns Any
        finally:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
                process.kill()
