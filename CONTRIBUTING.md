# Contributing to Hamiltonian VQE Kit (hvk)

Thank you for your interest in contributing to `hamiltonian-vqe-kit`! We welcome contributions adhering to rigorous scientific standards and honest reporting.

## Non-Negotiable Honesty Rules

1. **No Hardcoded Benchmarks**: Never invent, hardcode, or pre-write benchmark numbers. Every reported metric must stem from an automated script in `benchmarks/` saving full metadata (git commit, versions, random seeds).
2. **PySCF & FCI Validation**: Chemistry algorithms must validate numerically against PySCF Hartree-Fock and FCI exact ground states.
3. **Core is Non-ML**: Core functionality must run with zero ML dependencies. ML modules are strictly optional.
4. **Honest Baselines**: ML components must be benchmarked against exact non-ML and simple heuristic baselines. Null results must be reported transparently.
5. **Coverage & Quality**: All public functions require type hints, docstrings, unit tests, and coverage > 85%.

## Development Setup

```bash
# 1. Clone repository
git clone https://github.com/aashiq-parinda/hamiltonian-vqe-kit.git
cd hamiltonian-vqe-kit

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode with development dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

## Running Checks Locally

Before submitting a Pull Request, verify that all checks pass:

```bash
# Run linting
ruff check .

# Run formatting check
ruff format --check .

# Run type checker
mypy src tests

# Run test suite with coverage
pytest --cov=hvk --cov-report=term-missing
```

## Code Style

- Code must conform to Ruff formatting (quote style: double, line length: 100).
- Strict type hints are enforced via Mypy (`disallow_untyped_defs = true`).
