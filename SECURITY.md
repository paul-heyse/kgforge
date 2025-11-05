# Security & Supply Chain Hardening Report

## Vulnerability Scan Results

### pip-audit Summary

**Date**: Generated during Phase 2 R15 implementation
**Command**: `uv run pip-audit --format json`

**Results**: 1 vulnerability found

#### Vulnerability Details

- **Package**: `py` version 1.11.0
- **Vulnerability ID**: PYSEC-2022-42969
- **CVE**: CVE-2022-42969
- **GHSA**: GHSA-w596-4wvx-j9j6
- **Description**: The py library through 1.11.0 for Python allows remote attackers to conduct a ReDoS (Regular expression Denial of Service) attack via a Subversion repository with crafted info data, because the InfoSvnCommand argument is mishandled.
- **Severity**: Low (ReDoS, requires Subversion repository access)
- **Status**: Acceptable risk
  - `py` is a test dependency (pytest)
  - Vulnerability is in Subversion-related feature (`InfoSvnCommand`)
  - This feature is not used in the kgfoundry codebase
  - No direct user input flows through this code path
  - Mitigation: Monitor for updates; upgrade when available

### Recommendation

Monitor `py` package for updates. The vulnerability is in an unused code path and does not pose an immediate security risk to the application.

---

## Unsafe Constructs Review

### pickle Usage

**Status**: Documented and minimized

All `pickle.load` usage is:
1. Marked with `# noqa: S301` (security warning suppression)
2. Documented with comments indicating backward compatibility or trusted sources
3. Limited to legacy format support for migration purposes

**Locations**:
- `src/search_api/bm25_index.py`: Legacy pickle format support (line 371)
- `src/embeddings_sparse/bm25.py`: Legacy pickle format support (lines 276, 284)
- `src/embeddings_sparse/splade.py`: Legacy pickle format support (lines 306, 314)

**Migration Status**: In progress (R3 - Secure Serialization & Persistence)
- New code uses JSON with schema validation
- Legacy pickle files are being migrated to JSON format
- Backward compatibility maintained during transition

### YAML Loading

**Status**: ✅ Safe

All YAML loading uses `yaml.safe_load` (not `yaml.load`):
- `src/kgfoundry_common/config.py`: Uses `yaml.safe_load` (line 80)
- `tools/docs/scan_observability.py`: Uses `yaml.safe_load` (line 261)

### eval/exec Usage

**Status**: ✅ None found

Static analysis confirms no `eval` or `exec` calls in source code.

---

## Input Sanitization & Path Traversal Prevention

### Path Traversal Prevention

**Implementation**: `kgfoundry_common.fs.safe_join()`

- Validates that resolved paths stay within base directory
- Raises `ValueError` if path escapes base directory
- Requires absolute base path
- Prevents `..`, absolute paths, and symlink traversal attacks

**Test Coverage**: `tests/unit/test_security_helpers.py`
- Tests path traversal prevention
- Tests nested path handling
- Tests absolute base requirement

### Schema Validation

**Implementation**: Pydantic models with `ConfigDict(extra="forbid")`

- `SearchRequest`: Rejects extra fields
- `SearchResult`: Rejects extra fields
- `Doc`: Rejects extra fields

**Test Coverage**: `tests/unit/test_security_helpers.py`
- Tests rejection of extra fields
- Tests type validation
- Tests range validation

---

## Dependency Security Posture

### Summary

- **Total dependencies scanned**: ~800+
- **High-severity vulnerabilities**: 0
- **Medium-severity vulnerabilities**: 0
- **Low-severity vulnerabilities**: 1 (acceptable risk)

### Recommendations

1. **Continue monitoring**: Run `pip-audit` regularly (e.g., in CI)
2. **Upgrade `py`**: Monitor for updates to resolve PYSEC-2022-42969
3. **Document exceptions**: Any acceptable vulnerabilities should be documented with rationale
4. **Automate scanning**: Integrate `pip-audit` into CI/CD pipeline

---

## Test Results

All security regression tests pass:
- ✅ Input sanitization enforced
- ✅ Path traversal prevention verified
- ✅ Schema validation verified
- ✅ Unsafe constructs documented

**Test file**: `tests/unit/test_security_helpers.py`
**Test count**: 10 tests
**Status**: All passing

