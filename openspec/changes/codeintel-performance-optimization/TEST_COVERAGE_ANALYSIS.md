# FAISS Test Coverage Analysis

## Comparison: Our Tests vs FAISS Test Plan

### ✅ What We Have (Covered)

1. **Adaptive Index Selection** ✅
   - Tests for small/medium/large corpus thresholds
   - Verifies Flat/IVFFlat/IVF-PQ selection
   - Tests nlist calculation

2. **Memory Estimation** ✅
   - Tests for all three index types
   - Accuracy validation

3. **Basic Performance Benchmarks** ✅
   - Training time comparisons
   - Gated behind `RUN_BENCHMARKS` env var

### ❌ What's Missing (Critical Gaps)

#### A) Pre-flight Checks (Meta/FAISS Recommendations)

**Missing:**
- [ ] Vector hygiene checks (dtype float32, C-contiguous, no NaN/Inf)
- [ ] Unit normalization verification (for cosine/IP metric)
- [ ] Query dimension matching validation
- [ ] Index configuration sanity checks (nlist range, nprobe ≤ nlist)
- [ ] Training set size validation (≥ max(100k, 30×nlist))
- [ ] Ground-truth baseline (CPU Flat index for accuracy comparison)
- [ ] GPU environment detection and resource setup

**Our Syntax Issue:**
- Benchmarks use `_rng.standard_normal()` but should normalize vectors
- Unit tests correctly use `_rng.normal(0.5, 0.15)` + `np.clip()` + normalization
- **Fix needed**: Update benchmarks to match unit test vector generation

#### B) Basic Correctness Tests

**Missing:**
- [ ] **Flat exactness**: Compare `search()` results to NumPy brute-force
- [ ] **GPU Flat exactness**: GPU results match CPU Flat at fp32 tolerance
- [ ] **IVF-Flat correctness**: Recall@k vs nprobe sweeps (monotone curve)
- [ ] **IVFPQ correctness**: Reconstruction error validation
- [ ] **Filters & selectors**: `IDSelectorRange`, `IDSelectorBatch` (not implemented in FAISSManager)
- [ ] **Range search**: `range_search()` (not implemented in FAISSManager)
- [ ] **Serialization roundtrip**: `write_index` → `read_index` preserves results
- [ ] **Deletions**: `remove_ids()` (not implemented in FAISSManager)
- [ ] **Sharding/merging**: Multiple index merging (not implemented)

**Our Syntax Issues:**
- No ground-truth baseline comparison
- No recall@k calculations
- No serialization roundtrip tests

#### C) Robustness & Edge Cases

**Missing:**
- [ ] Untrained index errors (calling `add/search` before `train()`)
- [ ] Dimension mismatch errors
- [ ] k > ntotal handling
- [ ] OOM handling (GPU)
- [ ] Concurrency tests (multi-threaded search)
- [ ] Duplicate IDs handling (we have this in `update_index()` but no test)
- [ ] NaN/Inf input rejection

#### D) Accuracy & Performance Benchmarks

**Missing:**
- [ ] **Accuracy curves**: Recall@{1,5,10} vs latency/QPS
- [ ] **Build & memory profiles**: Train time, add time, index size
- [ ] **Scaling**: Thread scaling (1, 2, 4, 8, 16), batch size scaling
- [ ] **Stability**: Repeat runs, stddev < 10%

#### E) New Functions (Incremental Updates) - **NOT TESTED AT ALL**

**Missing:**
- [ ] `update_index()` - No tests
- [ ] `merge_indexes()` - No tests
- [ ] `_extract_all_vectors()` - No tests
- [ ] Dual-index search (`_search_primary`, `_search_secondary`, `_merge_results`) - No tests
- [ ] Secondary index creation and management - No tests

### 🔧 Syntax Corrections Needed

1. **Benchmark vector generation** (test_faiss_performance.py):
   ```python
   # CURRENT (WRONG):
   vectors = _rng.standard_normal((n_vectors, vec_dim)).astype(np.float32)
   
   # SHOULD BE (like unit tests):
   vectors = _rng.normal(0.5, 0.15, (n_vectors, vec_dim)).astype(np.float32)
   vectors = np.clip(vectors, 0.0, 1.0)
   # Then normalize in build_index (already done)
   ```

2. **Ground-truth baseline**: Need to add fixture that builds CPU Flat index and computes truth results

3. **Recall calculations**: Need helper functions to compute recall@k

### 📋 Recommended Test Structure

```
tests/codeintel_rev/faiss/
  conftest.py                    # Fixtures: rng, dims, truth baseline, gpu_resources
  test_preflight.py              # Meta/FAISS pre-checks
  test_flat_exactness.py         # CPU/GPU Flat vs NumPy brute-force
  test_ivf_recall.py             # Recall@k vs nprobe sweeps
  test_serialization.py          # Roundtrip tests
  test_incremental_updates.py    # update_index, merge_indexes, dual-index search
  test_robustness.py             # Edge cases, errors
  perf/
    bench_accuracy.py            # Recall@k vs latency/QPS
    bench_build_mem.py            # Build time, memory profiles
    bench_scaling.py              # Thread/batch scaling
```

### 🎯 Priority Actions

1. **HIGH**: Fix benchmark vector generation syntax
2. **HIGH**: Add tests for incremental update functions (update_index, merge_indexes, dual-index search)
3. **MEDIUM**: Add ground-truth baseline fixture
4. **MEDIUM**: Add recall@k calculation helpers and tests
5. **MEDIUM**: Add serialization roundtrip tests
6. **LOW**: Add pre-flight checks (can be part of conftest.py)
7. **LOW**: Add robustness/edge case tests

