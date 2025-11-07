"""Generate CodeIntel MCP tools documentation from server schemas."""

from __future__ import annotations

from pathlib import Path

from codeintel.mcp_server.server import MCPServer

HEADER = """# CodeIntel MCP Tools

Auto-generated from server schemas. Do not edit manually.

"""


def main() -> None:
    """Generate tools.md from MCPServer tool schemas."""
    server = MCPServer()
    tools_list = server._tool_schemas()
    lines = [HEADER]
    for tool in tools_list:
        name = tool["name"]
        description = tool.get("description", "")
        lines.append(f"## `{name}`\n\n{description}\n")
        schema = tool["inputSchema"]
        props = schema.get("properties", {})
        req = set(schema.get("required", []))
        if props:
            lines.append("**Parameters**:\n")
            for key, value in props.items():
                typ = value.get("type", "any")
                desc = value.get("description", "")
                star = " *(required)*" if key in req else ""
                lines.append(f"- `{key}`: `{typ}`{star} — {desc}")
            lines.append("\n")
        else:
            lines.append("**Parameters**: None\n\n")
    output_path = (
        Path(__file__).resolve().parents[3] / "docs" / "modules" / "codeintel" / "tools.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
