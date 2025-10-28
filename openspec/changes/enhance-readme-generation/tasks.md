# Implementation Tasks

## 1. Documentation

- [x] 1.1 Create `docs/how-to/read-package-readmes.md` contributor guide
  - [x] 1.1.1 Explain README structure (H1, synopsis, TOC, sections)
  - [x] 1.1.2 Document badge meanings (stability, owner, section, since, deprecated, tested-by)
  - [x] 1.1.3 Explain link types (open = editor, view = GitHub permalink)
  - [x] 1.1.4 Describe update workflow (when to regenerate, how to run)
  - [x] 1.1.5 Provide examples of each badge and link type
  - [x] 1.1.6 Document CLI flags and environment variables
  - [x] 1.1.7 Add troubleshooting section (missing metadata, broken links)

- [x] 1.2 Update `README-AUTOMATED-DOCUMENTATION.md`
  - [x] 1.2.1 Add "README Generation" section in Table of Contents
  - [x] 1.2.2 Document `tools/gen_readmes.py` in "Detailed Tool Descriptions"
  - [x] 1.2.3 Add README workflow to "Common Workflows"
  - [x] 1.2.4 Reference new contributor guide
  - [x] 1.2.5 Document relationship between NavMap/TestMap and READMEs

- [x] 1.3 Add inline code documentation
  - [x] 1.3.1 Add module-level docstring explaining README generation system
  - [x] 1.3.2 Document expected NavMap JSON structure in `_lookup_nav` docstring
  - [x] 1.3.3 Document expected TestMap JSON structure in `badges_for` docstring
  - [x] 1.3.4 Add examples to key functions (render_line, write_readme)

## 2. Code Refinements

- [x] 2.1 Enhance synopsis display
  - [x] 2.1.1 Verify `summarize(node)` extracts first sentence from `__init__` docstring
  - [x] 2.1.2 Ensure synopsis appears immediately after H1 before TOC markers
  - [x] 2.1.3 Add fallback message when synopsis is missing
  - [x] 2.1.4 Test synopsis extraction with various docstring formats

- [x] 2.2 Improve badge layout
  - [x] 2.2.1 Measure rendered badge line length in `format_badges`
  - [x] 2.2.2 Add newline after summary when badges exceed 80 chars
  - [x] 2.2.3 Indent continuation badges with 4 spaces
  - [x] 2.2.4 Test with symbols having all badges (stability/owner/section/since/deprecated/tested-by)

- [x] 2.3 Add missing file warnings
  - [x] 2.3.1 Check if `NAVMAP_PATH` exists in `main()`
  - [x] 2.3.2 Print warning (not error) if navmap.json missing: "Warning: NavMap not found at {path}; badges will be empty"
  - [x] 2.3.3 Check if `TESTMAP_PATH` exists in `main()`
  - [x] 2.3.4 Print warning if test_map.json missing: "Warning: Test map not found at {path}; tested-by badges will be empty"
  - [x] 2.3.5 Continue generation even when files are missing (graceful degradation)

- [x] 2.4 Add Doctoc integration
  - [x] 2.4.1 Add `--run-doctoc` boolean flag to argparse
  - [x] 2.4.2 Check if `doctoc` command exists (`shutil.which("doctoc")`)
  - [x] 2.4.3 After each README write, optionally run `doctoc <readme_path>`
  - [x] 2.4.4 Capture and log doctoc output if `--verbose` is enabled
  - [x] 2.4.5 Gracefully skip if doctoc not installed (print info message)
  - [x] 2.4.6 Test that TOC markers are populated correctly

## 3. Testing

- [x] 3.1 Create `tests/unit/test_gen_readmes.py`
  - [x] 3.1.1 Add imports and test fixtures (mock Griffe objects)
  - [x] 3.1.2 Create mock NavMap JSON structure
  - [x] 3.1.3 Create mock TestMap JSON structure

- [x] 3.2 Test determinism
  - [x] 3.2.1 Generate README twice with identical input
  - [x] 3.2.2 Assert both outputs are byte-for-byte identical
  - [x] 3.2.3 Verify provenance footer hash is stable
  - [x] 3.2.4 Test with various package structures (with/without subpackages)

- [x] 3.3 Test badge rendering
  - [x] 3.3.1 Test symbol with all badges present → all appear in output
  - [x] 3.3.2 Test symbol with no metadata → no badges appear
  - [x] 3.3.3 Test symbol with partial metadata → only present badges appear
  - [x] 3.3.4 Test badge ordering is consistent (stability, owner, section, since, deprecated, tested-by)
  - [x] 3.3.5 Test tested-by badge with multiple test files → shows top 3

- [x] 3.4 Test link generation
  - [x] 3.4.1 Test `--link-mode github` → only [view] links appear
  - [x] 3.4.2 Test `--link-mode editor` → only [open] links appear  
  - [x] 3.4.3 Test `--link-mode both` → both links appear
  - [x] 3.4.4 Test `--editor vscode` → generates `vscode://file/...` format
  - [x] 3.4.5 Test `--editor relative` → generates `./path:line:col` format
  - [x] 3.4.6 Validate GitHub permalink format: `https://github.com/{org}/{repo}/blob/{sha}/{path}#L{start}-L{end}`
  - [x] 3.4.7 Validate VSCode URL format: `vscode://file/{abs_path}:{line}:1`

- [x] 3.5 Test bucketing logic
  - [x] 3.5.1 Test module → appears in "Modules" section
  - [x] 3.5.2 Test package → appears in "Modules" section
  - [x] 3.5.3 Test regular class → appears in "Classes" section
  - [x] 3.5.4 Test exception class (inherits Exception) → appears in "Exceptions" section
  - [x] 3.5.5 Test exception class (name ends with Error) → appears in "Exceptions" section
  - [x] 3.5.6 Test function → appears in "Functions" section
  - [x] 3.5.7 Test private symbols (starting with _) → do not appear

- [x] 3.6 Test metadata missing behavior
  - [x] 3.6.1 Test `--fail-on-metadata-miss` with complete metadata → exits 0
  - [x] 3.6.2 Test `--fail-on-metadata-miss` with missing owner → exits 2, prints error
  - [x] 3.6.3 Test `--fail-on-metadata-miss` with missing stability → exits 2, prints error
  - [x] 3.6.4 Test default (no flag) with missing metadata → exits 0, no error

- [x] 3.7 Test dry-run mode
  - [x] 3.7.1 Test `--dry-run` → prints planned writes
  - [x] 3.7.2 Test `--dry-run` → does not modify any files
  - [x] 3.7.3 Verify file content unchanged after dry-run

## 4. CI Integration

- [x] 4.1 Add README check to `.github/workflows/ci.yml`
  - [x] 4.1.1 Add step after documentation build: "Check README freshness"
  - [x] 4.1.2 Run `python tools/gen_readmes.py --link-mode github --editor relative`
  - [x] 4.1.3 Check for git diff in `src/**/README.md` files
  - [x] 4.1.4 If diff found, print error message with regeneration command
  - [x] 4.1.5 Exit with code 1 if diff detected
  - [x] 4.1.6 Test CI check passes on clean tree
  - [x] 4.1.7 Test CI check fails when README is stale

- [x] 4.2 Document CI check in `README-AUTOMATED-DOCUMENTATION.md`
  - [x] 4.2.1 Add CI check to "Integration Points > CI/CD Integration" section
  - [x] 4.2.2 Explain what happens when check fails
  - [x] 4.2.3 Provide recovery command for developers

## 5. Pre-commit Hook (Optional)

- [x] 5.1 Create pre-commit hook configuration
  - [x] 5.1.1 Add hook entry in `.pre-commit-config.yaml` under `local` repo
  - [x] 5.1.2 Name: "readme-generator (changed packages only)"
  - [x] 5.1.3 Entry: bash script to detect changed packages from git diff
  - [x] 5.1.4 Pass detected packages to `DOCS_PKG` environment variable
  - [x] 5.1.5 Run `python tools/gen_readmes.py --link-mode github --editor relative`
  - [x] 5.1.6 Set `language: system`, `pass_filenames: false`
  - [x] 5.1.7 Comment the hook as "optional" in config

- [x] 5.2 Test pre-commit hook
  - [x] 5.2.1 Install pre-commit hooks (`pre-commit install`)
  - [x] 5.2.2 Modify a file in `src/kgfoundry_common/`
  - [x] 5.2.3 Stage change and attempt commit
  - [x] 5.2.4 Verify hook detects changed package
  - [x] 5.2.5 Verify hook runs gen_readmes only for that package
  - [x] 5.2.6 Verify hook updates README.md if needed

- [x] 5.3 Document pre-commit hook
  - [x] 5.3.1 Add to "Pre-Commit Hooks" section in `README-AUTOMATED-DOCUMENTATION.md`
  - [x] 5.3.2 Mark as optional with explanation of when to use it
  - [x] 5.3.3 Document performance characteristics (fast for single package)

## 6. Quality Assurance

- [x] 6.1 Run full pipeline on clean tree
  - [x] 6.1.1 Build NavMap: `python tools/navmap/build_navmap.py`
  - [x] 6.1.2 Build TestMap: `python tools/docs/build_test_map.py` (if exists)
  - [x] 6.1.3 Generate READMEs: `python tools/gen_readmes.py`
  - [x] 6.1.4 Verify no git diff produced
  - [x] 6.1.5 Verify all badges appear where metadata exists

- [x] 6.2 Test with missing metadata
  - [x] 6.2.1 Remove metadata from one symbol in navmap.json
  - [x] 6.2.2 Regenerate READMEs
  - [x] 6.2.3 Verify badges missing for that symbol only
  - [x] 6.2.4 Test `--fail-on-metadata-miss` catches the missing metadata

- [x] 6.3 Test across all packages
  - [x] 6.3.1 Run `DOCS_PKG=all python tools/gen_readmes.py --verbose`
  - [x] 6.3.2 Verify READMEs generated for all detected packages
  - [x] 6.3.3 Check for any errors or warnings in output
  - [x] 6.3.4 Verify provenance footers present in all READMEs

- [x] 6.4 Validate generated links
  - [x] 6.4.1 Click sample VSCode links → verify they open in editor
  - [x] 6.4.2 Click sample GitHub view links → verify they go to correct line
  - [x] 6.4.3 Verify line numbers are accurate (start and end)
  - [x] 6.4.4 Test with symbols at top and bottom of files

## 7. Documentation Review

- [x] 7.1 Review contributor guide
  - [x] 7.1.1 Have another developer read `docs/how-to/read-package-readmes.md`
  - [x] 7.1.2 Verify all sections are clear and actionable
  - [x] 7.1.3 Test that examples are accurate and runnable
  - [x] 7.1.4 Add screenshots if helpful (badge examples, link types)

- [x] 7.2 Update main README
  - [x] 7.2.1 Add one-line reference to README generation in root `README.md`
  - [x] 7.2.2 Link to full documentation in `README-AUTOMATED-DOCUMENTATION.md`

- [x] 7.3 Add command reference
  - [x] 7.3.1 Create "Quick Commands" section in contributor guide
  - [x] 7.3.2 List common usage patterns with copy-paste examples
  - [x] 7.3.3 Document troubleshooting commands

## 8. Final Validation

- [x] 8.1 Run all tests
  - [x] 8.1.1 `pytest tests/unit/test_gen_readmes.py -v`
  - [x] 8.1.2 Verify 100% pass rate
  - [x] 8.1.3 Check test coverage for `tools/gen_readmes.py` (aim for >90%)

- [x] 8.2 Run pre-commit checks
  - [x] 8.2.1 `pre-commit run --all-files`
  - [x] 8.2.2 Fix any linting or formatting issues
  - [x] 8.2.3 Verify navmap checks pass

- [x] 8.3 Run CI simulation locally
  - [x] 8.3.1 Execute CI README check command locally
  - [x] 8.3.2 Verify it passes on clean tree
  - [x] 8.3.3 Modify a file, regenerate, verify check still passes

- [x] 8.4 Documentation completeness check
  - [x] 8.4.1 Verify all new features documented
  - [x] 8.4.2 Verify all CLI flags explained
  - [x] 8.4.3 Verify examples provided for all features
  - [x] 8.4.4 Cross-check implementation matches specification

## Success Criteria

✅ **All tasks marked complete**
✅ **All tests passing** (`pytest tests/unit/test_gen_readmes.py`)
✅ **CI check passing** (README freshness validated)
✅ **Documentation complete** (contributor guide + main docs updated)
✅ **Deterministic output** (running twice produces no diff)
✅ **Links functional** (VSCode and GitHub links work correctly)
✅ **Badges accurate** (metadata correctly displayed)
✅ **Graceful degradation** (works even when NavMap/TestMap missing)

