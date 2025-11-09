# Contributing to Tensor Analysis Studio

Thank you for your interest in contributing to Tensor Analysis Studio!

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tensor-analysis-studio.git
   cd tensor-analysis-studio
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tas --cov-report=html

# Run specific test file
pytest tests/test_tensor.py -v

# Run tests matching pattern
pytest -k "test_einsum"
```

### Type Checking

```bash
mypy tas
```

### Code Formatting

```bash
# Format code
black tas tests

# Check formatting
black --check tas tests
```

### Linting

```bash
# Lint code
ruff check tas tests

# Auto-fix issues
ruff check --fix tas tests
```

## Code Style Guidelines

- **Type hints**: All public APIs must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public functions/classes
- **Line length**: Maximum 100 characters
- **Naming**: 
  - Functions/methods: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: prefix with `_`

### Example Docstring

```python
def function_name(arg1: int, arg2: str) -> bool:
    """
    Short description of function.
    
    Longer description providing more context and details
    about what the function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why this is raised
        
    Examples:
        >>> function_name(42, "test")
        True
    """
    pass
```

## Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach recommended)
   - Add tests in `tests/test_*.py`
   - Ensure tests fail initially

3. **Implement the feature**
   - Follow existing code patterns
   - Add type hints
   - Add docstrings

4. **Run tests and checks**
   ```bash
   pytest
   mypy tas
   black tas tests
   ruff check tas tests
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add description of feature"
   ```
   
   Use conventional commit messages:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` adding/updating tests
   - `refactor:` code refactoring
   - `perf:` performance improvements

6. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
tensor-analysis-studio/
├── tas/                    # Main package
│   ├── core/              # Core tensor functionality
│   │   ├── tensor.py      # Tensor class
│   │   ├── indices.py     # Index management
│   │   ├── einsum_parser.py  # Einstein notation
│   │   ├── metrics.py     # Metric tensors
│   │   ├── algebra.py     # Tensor operations
│   │   ├── connections.py # Christoffel symbols
│   │   ├── diffops.py     # Differential operators
│   │   └── coords.py      # Coordinate systems
│   ├── sym/               # Symbolic computation (optional)
│   └── viz/               # Visualization
├── tests/                 # Test suite
├── examples/              # Example notebooks and scripts
└── docs/                  # Documentation
```

## Testing Guidelines

- **Coverage**: Aim for >90% code coverage
- **Property-based tests**: Use Hypothesis for testing invariants
- **Fixtures**: Put common test data in `conftest.py`
- **Markers**: Use pytest markers for slow/integration tests

## Pull Request Process

1. Update documentation if adding new features
2. Add tests for new functionality
3. Ensure all tests pass and coverage doesn't decrease
4. Update CHANGELOG.md (if applicable)
5. Request review from maintainers
6. Address review feedback
7. Squash commits if requested

## Reporting Bugs

Use GitHub Issues with the bug template:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, package version
- Minimal code example

## Feature Requests

Use GitHub Issues with the feature template:
- Clear use case description
- Proposed API (if applicable)
- Why existing functionality doesn't suffice
- Willingness to contribute implementation

## Questions?

- Open a GitHub Discussion
- Check existing issues and documentation
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
