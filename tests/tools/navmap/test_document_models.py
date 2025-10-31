from __future__ import annotations

import msgspec
from tools import validate_tools_payload
from tools.navmap.document_models import NAVMAP_SCHEMA, navmap_document_from_index
from tools.navmap.models import (
    ModuleEntry,
    ModuleMeta,
    NavIndex,
    NavSection,
    SymbolMeta,
)


def test_navmap_document_round_trip_validates_schema() -> None:
    module_entry = ModuleEntry(
        path="pkg/example.py",
        exports=["Foo"],
        sections=[NavSection(id="public-api", symbols=["Foo"])],
        section_lines={"public-api": 10},
        anchors={"Foo": 15},
        links={"source": "./pkg/example.py"},
        meta={
            "Foo": SymbolMeta(
                owner="team-core", stability="stable", since="1.0.0", deprecated_in="2.0.0"
            )
        },
        module_meta=ModuleMeta(
            owner="team-core",
            stability="stable",
            since="1.0.0",
            deprecated_in="2.0.0",
        ),
        tags=["navmap"],
        synopsis="Example module",
        see_also=["pkg.other"],
        deps=["pkg.dep"],
    )
    index = NavIndex(
        commit="HEAD",
        policy_version="1",
        link_mode="editor",
        modules={"pkg.example": module_entry},
    )

    document = navmap_document_from_index(
        index,
        commit="HEAD",
        policy_version="1",
        link_mode="editor",
    )
    payload = msgspec.to_builtins(document)

    validate_tools_payload(payload, NAVMAP_SCHEMA)
    module_payload = payload["modules"]["pkg.example"]
    assert module_payload["moduleMeta"]["owner"] == "team-core"
    assert module_payload["meta"]["Foo"]["stability"] == "stable"
