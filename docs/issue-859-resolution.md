# Issue #859 Resolution Comment

## Resolution

The issue has been resolved by removing the dependency on sklearn's internal random utilities.

**Changes:**
- Updated `pyproject.toml`: `scipy>=1.16.0`, `numpy>=1.25.2`, `statsmodels>=0.14.5`, `requires-python>=3.11`
- Removed `from sklearn.utils._random cimport our_rand_r` import from `causalml/inference/tree/_tree/_utils.pyx`
- Copied `our_rand_r` and `RAND_R_MAX` implementations locally with proper BSD-3-Clause attribution

**Root Cause:**
The TypeError occurred because Cython auto-imports ALL symbols when using `cimport`, including `DEFAULT_SEED` which had a signature change in sklearn 1.6+ (const qualifier added/removed). Even though we only needed `our_rand_r`, the signature mismatch caused import failures.

**Verification:**
- ✓ `import causalml.dataset` succeeds
- ✓ All tree-based modules import successfully
- ✓ Test suite passes (109/109 tests)
- ✓ No Cython signature errors

Tested with: Python 3.11.9, sklearn 1.7.0, scipy 1.17.0, numpy 2.1.3
