from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fake_griffe = ModuleType("griffe")


class _LoaderStub:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def load(self, name: str) -> SimpleNamespace:  # pragma: no cover - replaced in tests
        raise NotImplementedError


fake_griffe.Object = SimpleNamespace
fake_griffe.GriffeLoader = _LoaderStub

fake_loader_module = ModuleType("griffe.loader")
fake_loader_module.GriffeLoader = _LoaderStub

sys.modules.setdefault("griffe", fake_griffe)
sys.modules.setdefault("griffe.loader", fake_loader_module)

fake_detect_pkg = ModuleType("detect_pkg")
fake_detect_pkg.detect_packages = lambda: []
fake_detect_pkg.detect_primary = lambda: "pkg"
sys.modules.setdefault("detect_pkg", fake_detect_pkg)

gr = importlib.import_module("tools.gen_readmes")


def _docstring(text: str | None) -> SimpleNamespace | None:
    if text is None:
        return None
    return SimpleNamespace(value=text)


def _node(
    *,
    path: str,
    kind: str,
    rel_path: str,
    summary: str | None = None,
    lineno: int = 1,
    endlineno: int | None = None,
    bases: list[SimpleNamespace] | None = None,
    members: dict[str, SimpleNamespace] | None = None,
    is_package: bool = False,
) -> SimpleNamespace:
    name = path.split(".")[-1]
    return SimpleNamespace(
        path=path,
        name=name,
        kind=SimpleNamespace(value=kind),
        docstring=_docstring(summary),
        relative_package_filepath=rel_path,
        lineno=lineno,
        endlineno=endlineno,
        bases=bases or [],
        members=members or {},
        is_package=is_package,
    )


@pytest.fixture()
def readme_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path
    src = root / "src"
    src.mkdir()
    monkeypatch.setattr(gr, "ROOT", root)
    monkeypatch.setattr(gr, "SRC", src)
    monkeypatch.setattr(gr, "OWNER", "acme")
    monkeypatch.setattr(gr, "REPO", "kgfoundry")
    monkeypatch.setattr(gr, "SHA", "deadbeefcafebabe")

    nav_path = root / "site" / "_build" / "navmap" / "navmap.json"
    nav_path.parent.mkdir(parents=True)
    nav_path.write_text("{}", encoding="utf-8")
    test_path = root / "docs" / "_build" / "test_map.json"
    test_path.parent.mkdir(parents=True)
    test_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(gr, "NAVMAP_PATH", nav_path)
    monkeypatch.setattr(gr, "TESTMAP_PATH", test_path)

    return {"root": root, "src": src, "nav": nav_path, "test": test_path}


def _package_tree(src: Path) -> SimpleNamespace:
    src.mkdir(parents=True, exist_ok=True)
    package_dir = src / "pkg"
    package_dir.mkdir()
    module_path = package_dir / "module.py"
    module_path.write_text("class Widget: ...\n", encoding="utf-8")

    func = _node(
        path="pkg.module.make_widget",
        kind="function",
        rel_path="pkg/module.py",
        summary="Create a widget.",
        lineno=10,
        endlineno=12,
    )
    cls = _node(
        path="pkg.module.Widget",
        kind="class",
        rel_path="pkg/module.py",
        summary="Widget container class.",
        lineno=20,
        endlineno=40,
    )
    exc = _node(
        path="pkg.module.WidgetError",
        kind="class",
        rel_path="pkg/module.py",
        summary="Raised when widget configuration fails.",
        lineno=42,
        endlineno=55,
        bases=[SimpleNamespace(name="Exception")],
    )
    module = _node(
        path="pkg.module",
        kind="module",
        rel_path="pkg/module.py",
        summary="Module utilities for widgets.",
        lineno=1,
        endlineno=80,
        members={
            "Widget": cls,
            "WidgetError": exc,
            "make_widget": func,
        },
    )
    pkg = _node(
        path="pkg",
        kind="package",
        rel_path="pkg/__init__.py",
        summary="Primary widget package. Additional docs ignored.",
        is_package=True,
        members={"module": module},
    )
    return pkg


def test_write_readme_is_deterministic(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _package_tree(readme_env["src"])
    monkeypatch.setattr(
        gr,
        "NAVMAP",
        {
            "modules": {
                "pkg.module": {
                    "meta": {
                        "pkg.module": {
                            "stability": "stable",
                            "owner": "@docs",
                            "section": "modules",
                        },
                        "pkg.module.Widget": {
                            "stability": "beta",
                            "owner": "@widgets",
                            "since": "1.2.0",
                            "deprecated_in": "2.0.0",
                        },
                        "pkg.module.make_widget": {"owner": "@widgets"},
                    },
                    "module_meta": {"stability": "experimental"},
                }
            }
        },
    )
    monkeypatch.setattr(
        gr,
        "TEST_MAP",
        {
            "pkg.module": [{"file": "tests/unit/test_module.py", "lines": [5]}],
            "pkg.module.Widget": [
                {"file": "tests/unit/test_widget.py", "lines": [42]},
                {"file": "tests/e2e/test_widget.py", "lines": [100]},
                {"file": "tests/api/test_widget.py", "lines": [12]},
                {"file": "tests/extra/test_widget.py", "lines": [1]},
            ],
        },
    )

    cfg = gr.Config(
        packages=["pkg"],
        link_mode="both",
        editor="vscode",
        fail_on_metadata_miss=False,
        dry_run=False,
        verbose=False,
        run_doctoc=False,
    )

    changed_first = gr.write_readme(pkg, cfg)
    assert changed_first is True
    content_1 = (readme_env["src"] / "pkg" / "README.md").read_text(encoding="utf-8")

    changed_second = gr.write_readme(pkg, cfg)
    assert changed_second is False
    content_2 = (readme_env["src"] / "pkg" / "README.md").read_text(encoding="utf-8")

    assert content_1 == content_2
    assert "## Modules" in content_1
    assert "`stability:stable`" in content_1
    assert "`owner:@docs`" in content_1
    assert "`section:modules`" in content_1
    assert "`tested-by: tests/unit/test_module.py:5`" in content_1
    assert "    `stability:stable`" in content_1


def test_format_badges_handles_partial_metadata(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        gr,
        "NAVMAP",
        {
            "modules": {
                "pkg.module": {
                    "meta": {"pkg.module.Widget": {"owner": "@docs"}},
                    "module_meta": {"stability": "stable"},
                }
            }
        },
    )
    monkeypatch.setattr(gr, "TEST_MAP", {})
    badge_text = gr.format_badges("pkg.module.Widget", base_length=10)
    assert badge_text.strip() == "`stability:stable` `owner:@docs`"


def test_render_line_respects_link_modes(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(
        path="pkg.module.func",
        kind="function",
        rel_path="pkg/module.py",
        summary="Make something.",
        lineno=12,
        endlineno=18,
    )
    pkg_dir = readme_env["src"] / "pkg"
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "module.py").write_text("def func(): ...\n", encoding="utf-8")
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})

    cfg_github = gr.Config(["pkg"], "github", "vscode", False, False, False, False)
    line = gr.render_line(node, pkg_dir, cfg_github)
    assert (
        "[view](https://github.com/acme/kgfoundry/blob/deadbeefcafebabe/src/pkg/module.py#L12-L18)"
        in line
    )
    assert "[open]" not in line

    cfg_editor = gr.Config(["pkg"], "editor", "relative", False, False, False, False)
    line_editor = gr.render_line(node, pkg_dir, cfg_editor)
    assert "[open](./src/pkg/module.py:12:1)" in line_editor
    assert "[view]" not in line_editor


def test_bucket_for_assignments() -> None:
    module = _node(path="pkg.mod", kind="module", rel_path="pkg/mod.py")
    package = _node(path="pkg", kind="package", rel_path="pkg/__init__.py", is_package=True)
    cls = _node(path="pkg.mod.Widget", kind="class", rel_path="pkg/mod.py")
    exc = _node(
        path="pkg.mod.WidgetError",
        kind="class",
        rel_path="pkg/mod.py",
        bases=[SimpleNamespace(name="Exception")],
    )
    func = _node(path="pkg.mod.build", kind="function", rel_path="pkg/mod.py")

    assert gr.bucket_for(module) == "Modules"
    assert gr.bucket_for(package) == "Modules"
    assert gr.bucket_for(cls) == "Classes"
    assert gr.bucket_for(exc) == "Exceptions"
    assert gr.bucket_for(func) == "Functions"


def test_render_line_wraps_badges(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    readme_dir = readme_env["src"] / "pkg"
    readme_dir.mkdir(exist_ok=True)
    (readme_dir / "widget.py").write_text("class Widget: ...\n", encoding="utf-8")
    node = _node(
        path="pkg.widget.Widget",
        kind="class",
        rel_path="pkg/widget.py",
        summary="Widget class summary sentence that is intentionally long to trigger wrapping.",
        lineno=5,
        endlineno=40,
    )
    monkeypatch.setattr(
        gr,
        "NAVMAP",
        {
            "modules": {
                "pkg.widget": {
                    "meta": {
                        "pkg.widget.Widget": {
                            "stability": "stable",
                            "owner": "@widgets",
                            "section": "public-api",
                            "since": "1.2.0",
                            "deprecated_in": "2.0.0",
                        }
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        gr,
        "TEST_MAP",
        {
            "pkg.widget.Widget": [
                {"file": "tests/unit/test_widget.py", "lines": [10]},
                {"file": "tests/e2e/test_widget.py", "lines": [20]},
                {"file": "tests/api/test_widget.py", "lines": [30]},
            ]
        },
    )
    cfg = gr.Config(["pkg"], "both", "vscode", False, False, False, False)
    line = gr.render_line(node, readme_dir, cfg)
    assert "\n    `stability:stable`" in line
    assert "tested-by" in line


def test_fail_on_metadata_miss_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pkg = _package_tree(tmp_path / "src")

    class Loader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load(self, name: str) -> SimpleNamespace:
            return pkg

    monkeypatch.setattr(gr, "GriffeLoader", Loader)
    monkeypatch.setattr(gr, "iter_packages", lambda: ["pkg"])
    monkeypatch.setattr(gr, "write_readme", lambda node, cfg: False)
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})
    monkeypatch.setattr(gr, "NAVMAP_PATH", Path("missing-nav.json"))
    monkeypatch.setattr(gr, "TESTMAP_PATH", Path("missing-test.json"))

    argv = ["gen_readmes", "--fail-on-metadata-miss"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        gr.main()
    assert exc.value.code == 2
    output = capsys.readouterr().out
    assert "Missing owner/stability" in output


def test_fail_on_metadata_miss_passes_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _package_tree(tmp_path / "src")

    class Loader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load(self, name: str) -> SimpleNamespace:
            return pkg

    monkeypatch.setattr(gr, "GriffeLoader", Loader)
    monkeypatch.setattr(gr, "iter_packages", lambda: ["pkg"])
    monkeypatch.setattr(gr, "write_readme", lambda node, cfg: False)
    monkeypatch.setattr(
        gr,
        "NAVMAP",
        {
            "modules": {
                "pkg.module": {
                    "meta": {
                        "pkg.module": {"owner": "@docs", "stability": "stable"},
                        "pkg.module.Widget": {"owner": "@docs", "stability": "stable"},
                        "pkg.module.WidgetError": {"owner": "@docs", "stability": "stable"},
                        "pkg.module.make_widget": {"owner": "@docs", "stability": "stable"},
                    }
                }
            }
        },
    )
    monkeypatch.setattr(gr, "TEST_MAP", {})
    monkeypatch.setattr(gr, "NAVMAP_PATH", tmp_path / "nav.json")
    monkeypatch.setattr(gr, "TESTMAP_PATH", tmp_path / "test.json")
    (tmp_path / "nav.json").write_text("{}", encoding="utf-8")
    (tmp_path / "test.json").write_text("{}", encoding="utf-8")

    argv = ["gen_readmes", "--fail-on-metadata-miss"]
    monkeypatch.setattr(sys, "argv", argv)
    badge_widget = gr.badges_for("pkg.module.Widget")
    badge_error = gr.badges_for("pkg.module.WidgetError")
    badge_func = gr.badges_for("pkg.module.make_widget")
    assert badge_widget.owner == "@docs" and badge_widget.stability == "stable"
    assert badge_error.owner == "@docs" and badge_error.stability == "stable"
    assert badge_func.owner == "@docs" and badge_func.stability == "stable"
    gr.main()


def test_missing_navmap_and_testmap_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pkg = _package_tree(tmp_path / "src")

    class Loader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load(self, name: str) -> SimpleNamespace:
            return pkg

    monkeypatch.setattr(gr, "GriffeLoader", Loader)
    monkeypatch.setattr(gr, "iter_packages", lambda: ["pkg"])
    monkeypatch.setattr(gr, "write_readme", lambda node, cfg: False)
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})
    monkeypatch.setattr(gr, "NAVMAP_PATH", tmp_path / "missing-nav.json")
    monkeypatch.setattr(gr, "TESTMAP_PATH", tmp_path / "missing-test.json")

    argv = ["gen_readmes"]
    monkeypatch.setattr(sys, "argv", argv)
    gr.main()
    output = capsys.readouterr().out
    assert "Warning: NavMap not found" in output
    assert "Warning: Test map not found" in output


def test_dry_run_reports_without_writing(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pkg = _package_tree(readme_env["src"])
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})
    cfg = gr.Config(["pkg"], "both", "vscode", False, True, False, False)
    changed = gr.write_readme(pkg, cfg)
    assert changed is False
    out = capsys.readouterr().out
    assert "[dry-run] would write" in out
    assert not (readme_env["src"] / "pkg" / "README.md").exists()


def test_package_synopsis_fallback(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _package_tree(readme_env["src"])
    pkg.docstring = None
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})
    cfg = gr.Config(["pkg"], "both", "vscode", False, False, False, False)
    gr.write_readme(pkg, cfg)
    content = (readme_env["src"] / "pkg" / "README.md").read_text(encoding="utf-8")
    assert gr.DEFAULT_SYNOPSIS in content


def test_doctoc_invoked_when_enabled(
    readme_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _package_tree(readme_env["src"])
    monkeypatch.setattr(gr, "NAVMAP", {})
    monkeypatch.setattr(gr, "TEST_MAP", {})

    calls: list[list[str]] = []

    def fake_which(command: str) -> str | None:
        return "/usr/bin/doctoc" if command == "doctoc" else None

    class Result:
        def __init__(self) -> None:
            self.stdout = "updated toc"
            self.stderr = ""
            self.returncode = 0

    def fake_run(cmd: list[str], **_: object) -> Result:
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(gr.shutil, "which", fake_which)
    monkeypatch.setattr(gr.subprocess, "run", fake_run)

    cfg = gr.Config(["pkg"], "both", "vscode", False, False, True, True)
    gr.write_readme(pkg, cfg)
    assert calls and Path(calls[0][-1]).name == "README.md"
