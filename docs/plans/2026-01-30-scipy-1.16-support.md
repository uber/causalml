# Scipy 1.16+ Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable causalml 0.15.6dev to support scipy>=1.16.0, numpy>=1.25.2, statsmodels>=0.14.5, and Python>=3.11

**Architecture:** The issue is a Cython binary interface incompatibility. When dependencies (scikit-learn, numpy, scipy) are upgraded, the Cython extensions compiled against older versions have mismatched type signatures. The solution is to rebuild all Cython extensions against the updated dependencies.

**Tech Stack:** Cython, scikit-learn>=1.6.0, numpy>=1.25.2, scipy>=1.16.0

**Root Cause:** The error `TypeError: C variable sklearn.utils._random.DEFAULT_SEED has wrong signature (expected __pyx_t_7sklearn_5utils_9_typedefs_uint32_t const , got __pyx_t_7sklearn_5utils_9_typedefs_uint32_t)` occurs because:
- Cython extensions in `causalml/inference/tree/` import from `sklearn.utils._random`
- The binary interface changed between scikit-learn versions
- Old compiled `.so`/`.pyd` files have incompatible type signatures

---

### Task 1: Clean Existing Build Artifacts

**Files:**
- Remove: `build/` directory
- Remove: `causalml.egg-info/` directory
- Remove: `dist/` directory
- Remove: All `.so` files in `causalml/inference/tree/`
- Remove: All `.c` files generated from `.pyx` files

**Step 1: Run clean command**

Run: `make clean`
Expected: Removes build/, dist/, *.egg-info/, and compiled Cython files

**Step 2: Verify .so files are removed**

Run: `find causalml/inference/tree -name "*.so" -o -name "*.pyd"`
Expected: No output (all compiled extensions removed)

**Step 3: Verify generated .c files are removed**

Run: `find causalml/inference/tree -name "*.c" | grep -E "(criterion|splitter|tree|utils|builder)"`
Expected: No output (all generated C files removed)

**Step 4: Commit clean state**

```bash
git status
# Should show no changes if .so/.c files are gitignored
```

---

### Task 2: Fix sklearn Import Incompatibility

**Root Cause:** sklearn 1.6+ changed `DEFAULT_SEED` signature from `const uint32_t` to `uint32_t`. When causalml imports from `sklearn.utils._random`, Cython auto-imports ALL public symbols including `DEFAULT_SEED`, causing a type signature mismatch even though we don't use it.

**Solution:** Copy `our_rand_r` implementation directly into causalml to avoid sklearn import.

**Files:**
- Modify: `causalml/inference/tree/_tree/_utils.pyx`

**Step 1: Remove sklearn import**

Remove line 19:
```cython
from sklearn.utils._random cimport our_rand_r
```

**Step 2: Add local random utility implementation**

Add after line 18 (after `cnp.import_array()`):

```cython
# Random number generation utilities
# Copied from sklearn.utils._random to avoid DEFAULT_SEED signature mismatch
# Original authors: The scikit-learn developers
# License: BSD-3-Clause

from ._typedefs cimport uint32_t

cdef inline uint32_t DEFAULT_SEED = 1

cdef enum:
    # Max value for our rand_r replacement.
    # Corresponds to the maximum representable value for
    # 32-bit signed integers (i.e. 2^31 - 1).
    RAND_R_MAX = 2147483647

cdef inline uint32_t our_rand_r(uint32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0):
        seed[0] = DEFAULT_SEED

    seed[0] ^= <uint32_t>(seed[0] << 13)
    seed[0] ^= <uint32_t>(seed[0] >> 17)
    seed[0] ^= <uint32_t>(seed[0] << 5)

    # Use the modulo to ensure we don't return values greater than
    # the maximum representable value for signed 32bit integers.
    return seed[0] % ((<uint32_t>RAND_R_MAX) + 1)
```

**Step 3: Verify the edit**

Run: `grep -n "from sklearn.utils._random" causalml/inference/tree/_tree/_utils.pyx`
Expected: No output (import removed)

Run: `grep -n "our_rand_r" causalml/inference/tree/_tree/_utils.pyx`
Expected: Shows the new implementation and usage in rand_int/rand_uniform

**Step 4: Commit the fix**

```bash
git add causalml/inference/tree/_tree/_utils.pyx
git commit -m "fix: remove sklearn.utils._random import to avoid DEFAULT_SEED signature mismatch

- Copy our_rand_r and RAND_R_MAX implementations locally
- Avoids sklearn 1.6+ DEFAULT_SEED const qualifier change
- Maintains BSD-3-Clause license compatibility"
```

---

### Task 3: Rebuild Cython Extensions

**Files:**
- Build: All `.pyx` files in `causalml/inference/tree/`
  - `causalml/inference/tree/_tree/_tree.pyx`
  - `causalml/inference/tree/_tree/_criterion.pyx`
  - `causalml/inference/tree/_tree/_splitter.pyx`
  - `causalml/inference/tree/_tree/_utils.pyx`
  - `causalml/inference/tree/causal/_criterion.pyx`
  - `causalml/inference/tree/causal/_builder.pyx`
  - `causalml/inference/tree/uplift.pyx`

**Step 1: Clean build artifacts**

Run: `find causalml/inference/tree -name "*.so" -delete`
Expected: Removes old compiled extensions

**Step 2: Rebuild extensions**

Run: `uv pip install -e .`
Expected: Successful compilation with no errors, generates new .so/.pyd files

**Step 3: Verify compiled extensions exist**

Run: `find causalml/inference/tree -name "*.so" -o -name "*.pyd" | wc -l`
Expected: 7 (one for each .pyx file)

**Step 4: Check for compilation warnings**

Review the build output for deprecation warnings or errors
Expected: Clean build or only minor warnings

---

### Task 4: Verify Import Success

**Files:**
- Test: `causalml/dataset/__init__.py`
- Test: `causalml/inference/tree/__init__.py`

**Step 1: Test basic import**

Run:
```bash
uv run python -c "import causalml.dataset; print('✓ causalml.dataset imported successfully')"
```
Expected: `✓ causalml.dataset imported successfully`

**Step 2: Test tree imports**

Run:
```bash
uv run python -c "from causalml.inference.tree import CausalTreeRegressor; print('✓ CausalTreeRegressor imported successfully')"
```
Expected: `✓ CausalTreeRegressor imported successfully`

**Step 3: Test problematic import from issue #859**

Run:
```bash
uv run python -c "from sklearn.utils._random import DEFAULT_SEED; import causalml.dataset; print(f'✓ No signature error, DEFAULT_SEED={DEFAULT_SEED}')"
```
Expected: No TypeError, prints DEFAULT_SEED value

---

### Task 5: Run Test Suite

**Files:**
- Test: `tests/` directory

**Step 1: Run conftest loading test**

Run: `uv run pytest tests/conftest.py -v`
Expected: PASS or successful collection

**Step 2: Run quick smoke tests**

Run: `uv run pytest tests/test_causaltree.py -v -k "test_causaltree_regressor_fit" --maxfail=1`
Expected: PASS (at least one tree test passes)

**Step 3: Run full test suite (sample)**

Run: `uv run pytest tests/ -v --maxfail=5 -x`
Expected: Tests run without the Cython import error

Note: Some tests may fail for other reasons, but the Cython signature error should be resolved

**Step 4: Run meta-learner tests**

Run: `uv run pytest tests/test_meta_learners.py -v --maxfail=3`
Expected: Tests execute (may have failures unrelated to Cython)

---

### Task 6: Update Build Documentation

**Files:**
- Modify: `CLAUDE.md` (if needed)
- Modify: `README.md` (if needed)

**Step 1: Check if CLAUDE.md needs updates**

Review `CLAUDE.md` sections on:
- Environment Setup
- Build Commands
- Dependencies

**Step 2: Add note about dependency upgrades**

If not already documented, add to CLAUDE.md:
```markdown
### Dependency Upgrades

When upgrading major dependencies (scikit-learn, numpy, scipy):
1. Update version requirements in `pyproject.toml`
2. Clean build artifacts: `make clean`
3. Rebuild Cython extensions: `make build_ext` or `uv pip install -e .`
4. Run tests to verify: `uv run pytest tests/`
```

**Step 3: Commit documentation updates**

```bash
git add CLAUDE.md
git commit -m "docs: add dependency upgrade workflow to CLAUDE.md"
```

---

### Task 7: Verify Issue #859 Resolution

**Files:**
- Test: Exact reproduction from issue #859

**Step 1: Test with fresh environment simulation**

Run:
```bash
uv run python -c "
# Simulate the issue #859 reproduction
import causalml.dataset
print('✓ Issue #859 resolved: causalml.dataset imports without TypeError')
"
```
Expected: Success message

**Step 2: Document resolution in issue**

Prepare comment for issue #859:
```markdown
## Resolution

The issue has been resolved by rebuilding Cython extensions against the updated dependencies.

**Changes:**
- Updated `pyproject.toml`: `scipy>=1.16.0`, `numpy>=1.25.2`, `statsmodels>=0.14.5`, `requires-python>=3.11`
- Rebuilt all Cython extensions in `causalml/inference/tree/`

**Root Cause:**
The TypeError occurred because Cython-compiled extensions had binary interface mismatches with scikit-learn 1.6.0+. The signature of `sklearn.utils._random.DEFAULT_SEED` changed (const qualifier), causing type incompatibility.

**Verification:**
- ✓ `import causalml.dataset` succeeds
- ✓ All tree-based modules import successfully
- ✓ Tests run without Cython signature errors

**For users:** If upgrading to causalml 0.15.6+, you may need to reinstall:
```
pip uninstall causalml
pip install causalml --no-cache-dir
```
```

**Step 3: Update version and changelog**

If releasing, update version in `pyproject.toml` from `0.15.6dev` to `0.15.6` and add to CHANGELOG.

---

### Task 8: Final Integration Test

**Files:**
- Test: End-to-end functionality

**Step 1: Test causal tree workflow**

Run:
```bash
uv run python -c "
import numpy as np
from causalml.inference.tree import CausalTreeRegressor

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 5)
treatment = np.random.binomial(1, 0.5, 100)
y = X[:, 0] + treatment * X[:, 1] + np.random.randn(100) * 0.1

# Fit model
ct = CausalTreeRegressor()
ct.fit(X, treatment, y)
te = ct.predict(X)

print(f'✓ CausalTreeRegressor works: mean TE = {te.mean():.3f}')
"
```
Expected: Success with treatment effect estimate

**Step 2: Test meta-learner workflow**

Run:
```bash
uv run python -c "
import numpy as np
from causalml.inference.meta import BaseSRegressor
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)
X = np.random.randn(100, 5)
treatment = np.random.binomial(1, 0.5, 100)
y = X[:, 0] + treatment * 2 + np.random.randn(100) * 0.1

learner = BaseSRegressor(RandomForestRegressor())
learner.fit(X, treatment, y)
te = learner.predict(X)

print(f'✓ S-Learner works: mean TE = {te.mean():.3f}')
"
```
Expected: Success with treatment effect estimate

**Step 3: Commit final changes**

```bash
git status
git add -A
git commit -m "fix: rebuild Cython extensions for scipy>=1.16.0 support

Resolves #859

- Clean and rebuild all Cython extensions in causalml/inference/tree/
- Support scipy>=1.16.0, numpy>=1.25.2, statsmodels>=0.14.5
- Requires Python>=3.11
- Fixes TypeError with sklearn.utils._random.DEFAULT_SEED signature mismatch"
```

---

## Testing Checklist

- [ ] Build artifacts cleaned
- [ ] Cython extensions rebuilt successfully
- [ ] `import causalml.dataset` works
- [ ] Tree-based imports work
- [ ] Basic test suite runs (conftest loads)
- [ ] Causal tree end-to-end test passes
- [ ] Meta-learner end-to-end test passes
- [ ] No Cython signature errors in pytest output

## Success Criteria

1. All Cython extensions compile without errors
2. `import causalml.dataset` succeeds (resolves issue #859)
3. Test suite runs without Cython-related import errors
4. Basic functionality tests pass (tree and meta-learner workflows)

## Notes

- The `.so`/`.pyd` files should be in `.gitignore` (they are platform/version specific)
- Users upgrading from 0.15.5 to 0.15.6+ will need to rebuild or reinstall
- CI/CD pipelines should always do clean builds
- The issue affects Python 3.13+ with newest dependencies, but fix works for all versions
