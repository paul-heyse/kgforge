## MODIFIED Requirements

### Requirement: Package README Generation
The system SHALL generate comprehensive README.md files for each Python package with metadata badges, deep links, and deterministic content organization.

#### Scenario: Complete README generation with all metadata
- **WHEN** `python tools/gen_readmes.py` is executed with NavMap and TestMap JSON files present
- **THEN** each package README contains H1 with package name, package synopsis, TOC markers, grouped sections (Modules/Classes/Functions/Exceptions), symbols with one-line summaries, metadata badges (stability/owner/section/since/deprecated/tested-by), editor deep-links, and GitHub permalink view links

#### Scenario: README generation with missing metadata files
- **WHEN** `python tools/gen_readmes.py` is executed but `site/_build/navmap/navmap.json` or `docs/_build/test_map.json` does not exist
- **THEN** system prints warning messages indicating which files are missing, continues generation without errors, and produces READMEs with empty badge sections where metadata would appear

#### Scenario: Deterministic output prevents unnecessary diffs
- **WHEN** `python tools/gen_readmes.py` is executed twice consecutively without code changes
- **THEN** generated README content is byte-for-byte identical including provenance footer hash, and no git diff is produced

#### Scenario: Link mode configuration controls output format
- **WHEN** `python tools/gen_readmes.py --link-mode github` is executed
- **THEN** only [view] links to GitHub permalinks appear in README entries
- **WHEN** `python tools/gen_readmes.py --link-mode editor` is executed  
- **THEN** only [open] links (VSCode URLs or relative paths) appear in README entries
- **WHEN** `python tools/gen_readmes.py --link-mode both` is executed
- **THEN** both [open] and [view] links appear for each symbol

#### Scenario: VSCode deep-links enable direct editor navigation
- **WHEN** `python tools/gen_readmes.py --link-mode editor --editor vscode` is executed
- **THEN** [open] links use format `vscode://file/{absolute_path}:{line}:1` which opens files directly in VSCode when clicked

#### Scenario: Exception classes are grouped separately
- **WHEN** a Python class inherits from Exception or has name ending in Error/Exception
- **THEN** the class appears under "Exceptions" section not "Classes" section in generated README

#### Scenario: Badge layout remains readable with many badges
- **WHEN** a symbol has stability, owner, section, since, deprecated, and tested-by badges
- **THEN** badges wrap to new line after summary when total length exceeds 80 characters, with continuation lines indented 4 spaces

#### Scenario: Fail-fast validation for missing metadata
- **WHEN** `python tools/gen_readmes.py --fail-on-metadata-miss` is executed and any public symbol lacks owner or stability metadata
- **THEN** system prints error listing all symbols missing metadata, exits with code 2, and does not write any README files

#### Scenario: Dry-run mode previews changes without modification
- **WHEN** `python tools/gen_readmes.py --dry-run` is executed
- **THEN** system prints "[dry-run] would write {path}" messages for each README that would be modified, and does not actually write or modify any files

#### Scenario: Doctoc integration populates TOC automatically
- **WHEN** `python tools/gen_readmes.py --run-doctoc` is executed and doctoc command is installed
- **THEN** after writing each README, system runs doctoc to populate TOC between HTML comment markers, and TOC reflects current section structure

## ADDED Requirements

### Requirement: README Contributor Documentation
The system SHALL provide comprehensive documentation explaining README structure, badge meanings, link types, and update workflows for developers.

#### Scenario: Contributor guide explains README structure
- **WHEN** developer reads `docs/how-to/read-package-readmes.md`
- **THEN** guide explains H1 format, synopsis location, TOC marker purpose, section organization, badge meanings, link types, and includes examples

#### Scenario: Main documentation references README system
- **WHEN** developer reads `README-AUTOMATED-DOCUMENTATION.md`
- **THEN** document includes "README Generation" section with tool description, workflow examples, and reference to contributor guide

### Requirement: CI README Validation
The system SHALL validate README freshness in continuous integration to prevent stale documentation from being merged.

#### Scenario: CI detects stale READMEs
- **WHEN** CI runs after code change that should update README
- **THEN** CI step runs `python tools/gen_readmes.py --link-mode github --editor relative`, checks git diff for `src/**/README.md`, prints error with regeneration command if diff detected, and fails CI with exit code 1

#### Scenario: CI passes with up-to-date READMEs
- **WHEN** CI runs and all READMEs match current codebase
- **THEN** README check step completes successfully with no diff detected

### Requirement: Optional Pre-commit README Hook
The system SHALL provide an optional pre-commit hook that regenerates READMEs only for changed packages to maintain consistency without excessive overhead.

#### Scenario: Pre-commit hook detects changed package
- **WHEN** developer commits changes to files in `src/kgfoundry_common/`
- **THEN** pre-commit hook extracts "kgfoundry_common" from changed file paths, runs `DOCS_PKG=kgfoundry_common python tools/gen_readmes.py`, and updates README if needed before commit completes

#### Scenario: Pre-commit hook skipped when not installed
- **WHEN** pre-commit hooks are not installed via `pre-commit install`
- **THEN** README generation does not run automatically and developers must run manually

### Requirement: README Generation Test Coverage
The system SHALL include comprehensive tests validating determinism, badge rendering, link generation, bucketing logic, and metadata handling.

#### Scenario: Determinism test validates stable output
- **WHEN** test generates README twice with identical mock data
- **THEN** output is byte-for-byte identical including provenance footer hash

#### Scenario: Badge rendering test validates metadata display
- **WHEN** test provides symbol with all metadata fields populated
- **THEN** output includes all badges in correct order: stability, owner, section, since, deprecated, tested-by

#### Scenario: Link format test validates URL structure
- **WHEN** test generates links with `--editor vscode` and `--link-mode both`
- **THEN** VSCode URLs match pattern `vscode://file/{abs}:{line}:1` and GitHub URLs match pattern `https://github.com/{org}/{repo}/blob/{sha}/{path}#L{start}-L{end}`

#### Scenario: Bucketing test validates section assignment
- **WHEN** test provides module, class, exception, and function objects
- **THEN** each appears in correct section: Modules, Classes, Exceptions, Functions respectively

