# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CausalML is a Python package for uplift modeling and causal inference with machine learning algorithms. It provides methods to estimate Conditional Average Treatment Effect (CATE) or Individual Treatment Effect (ITE) from experimental or observational data.

## Development Setup

### Environment Setup
- Python 3.9+ required (supports 3.9-3.12)
- Uses `uv` as the package manager (preferred)
- Install development dependencies with `make setup_local` (sets up pre-commit hooks)

### Build Commands
- `uv pip install -e .`: Install package in editable mode (preferred)
- `make build_ext`: Build Cython extensions (required before running code/tests)
- `make build`: Build wheel distribution
- `make install`: Install package locally
- `make clean`: Clean build artifacts

### Testing
- `uv run pytest -vs --cov causalml/`: Run full test suite with coverage
- `uv run pytest tests/test_specific.py`: Run specific test file
- `make test`: Alternative full test suite command
- Optional test flags:
  - `uv run pytest --runtf`: Include TensorFlow tests
  - `uv run pytest --runtorch`: Include PyTorch tests

### Code Quality
- **ALWAYS run black before pushing**: Ensures code formatting compliance
  ```bash
  uv add black  # Add black if not already installed
  uv run black .  # Format all Python files
  ```
- Uses `black` for code formatting (max line length 120)
- Pre-commit hooks available via `make setup_local`
- Flake8 configuration in tox.ini

## Architecture

### Core Module Structure
```
causalml/
├── dataset/           # Synthetic data generation
├── feature_selection/ # Feature selection utilities
├── inference/         # Main inference algorithms
│   ├── meta/         # Meta-learners (S, T, X, R, DR learners)
│   ├── tree/         # Causal trees and uplift trees
│   ├── tf/           # TensorFlow implementations (DragonNet)
│   ├── torch/        # PyTorch implementations (CEVAE)
│   └── iv/           # Instrumental variable methods
├── metrics/          # Evaluation metrics
├── optimize/         # Policy learning and optimization
└── propensity.py     # Propensity score modeling
```

### Key Components

#### Meta-Learners (`causalml/inference/meta/`)
- **BaseLearner**: Abstract base class for all meta-learners
- **S-Learner**: Single model approach
- **T-Learner**: Two model approach
- **X-Learner**: Cross-learner with propensity scores
- **R-Learner**: Robinson's R-learner
- **DR-Learner**: Doubly robust learner

#### Tree-Based Methods (`causalml/inference/tree/`)
- Causal trees and forests with Cython implementations
- Uplift trees for classification problems
- Custom splitting criteria for causal inference

#### Propensity Score Models (`causalml/propensity.py`)
- **PropensityModel**: Abstract base for propensity estimation
- Built-in calibration support
- Clipping bounds to avoid numerical issues

### Cython Extensions
The package includes Cython-compiled modules for performance:
- Tree algorithms (`_tree`, `_criterion`, `_splitter`, `_utils`)
- Causal tree components (`_builder`, causal trees)
- Always run `make build_ext` after changes to .pyx files

## Common Workflows

### Adding New Meta-Learners
1. Inherit from `BaseLearner` in `causalml/inference/meta/base.py`
2. Implement `fit()` and `predict()` methods
3. Add appropriate tests in `tests/test_meta_learners.py`

### Working with Tree Methods
1. Cython files are in `causalml/inference/tree/`
2. Rebuild extensions with `make build_ext` after changes
3. Test with synthetic data from `causalml.dataset`

### Testing Different Backends
- Core tests run without optional dependencies
- TensorFlow tests: `uv run pytest --runtf`
- PyTorch tests: `uv run pytest --runtorch`
- Tests use fixtures from `tests/conftest.py` for data generation

### Complete Development Workflow
1. **Setup**: `uv pip install -e .` (install in editable mode)
2. **Development**: Make code changes
3. **Testing**: `uv run pytest tests/test_specific.py` (test your changes)
4. **Formatting**: `uv run black .` (REQUIRED before commit)
5. **Commit**: `git add -A && git commit -m "Your changes"`
6. **Push**: Use SSH key command below

### Git Operations
- **Before pushing any branch**: Always run code formatting
  ```bash
  uv run black .  # Format code
  git add -A      # Stage formatting changes (if any)
  git commit -m "Apply black formatting" # Only if files were changed
  ```
- **Pushing branches**: Use specific SSH key for authentication:
  ```bash
  GIT_SSH_COMMAND='ssh -i ~/.ssh/github_personal -o IdentitiesOnly=yes' git push -u origin branch_name
  ```

## Important Notes

- The package uses both pandas DataFrames and numpy arrays internally
- Propensity scores are clipped by default to avoid division by zero
- Meta-learners support both single and multiple treatment scenarios
- Tree methods include built-in visualization capabilities
- Optional dependencies (TensorFlow, PyTorch) are marked clearly in tests