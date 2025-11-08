# Contributing to Numeric Integrator

Thank you for considering contributing to this project! This guide will help you get started.

## Development Setup

1. **Clone and install:**
```bash
git clone https://github.com/yourusername/numeric-integrator.git
cd numeric-integrator
pip install -e ".[dev]"
```

2. **Install development dependencies:**
```bash
pip install pytest pytest-cov scipy matplotlib
```

3. **Run tests to verify setup:**
```bash
pytest tests/ -v
```

## Code Structure

```
numeric_integrator/
├── __init__.py          # Package exports
├── errors.py            # Custom exceptions
├── integrators.py       # Integration methods
├── differentiators.py   # Differentiation methods
├── ode_solvers.py       # ODE solvers
├── analysis.py          # Error analysis tools
├── utils.py             # Helper functions
└── api.py               # High-level interface

tests/
├── test_integrators.py
├── test_differentiators.py
├── test_ode_solvers.py
├── test_analysis.py
├── test_api.py
└── test_utils.py

examples/
├── integration_demo.ipynb
├── differentiation_demo.ipynb
├── ode_demo.ipynb
└── benchmark.py
```

## Contribution Guidelines

### Adding a New Integration Method

1. **Implement in `integrators.py`:**
```python
def your_method(f, a, b, n=100, **kwargs):
    """
    Brief description
    
    Parameters
    ----------
    f : callable
        Function to integrate
    a, b : float
        Integration bounds
    n : int
        Number of intervals
        
    Returns
    -------
    IntegrationResult
        Result with value, error, n_evaluations, method
    """
    # Implementation
    ...
    return IntegrationResult(value, error, n_evals, "your_method")
```

2. **Add to API in `api.py`:**
```python
def integrate(f, a, b, method='simpson', **kwargs):
    methods = {
        ...
        'your_method': lambda: your_method(f, a, b, **kwargs)
    }
```

3. **Write tests in `tests/test_integrators.py`:**
```python
class TestYourMethod:
    def test_polynomial(self):
        # Test on x^2
        result = your_method(lambda x: x**2, 0, 1, n=100)
        assert abs(result.value - 1/3) < 1e-6
        
    def test_convergence(self):
        # Verify convergence rate
        ...
```

4. **Add example to documentation**

### Adding a New ODE Solver

1. **Implement in `ode_solvers.py`:**
```python
def your_solver(f, y0, x0, x_end, step=0.1, **kwargs):
    """
    Brief description with order and stability info
    
    Parameters
    ----------
    f : callable
        Right-hand side f(x, y)
    y0 : float or array
        Initial condition
    x0, x_end : float
        Integration interval
    step : float
        Step size
        
    Returns
    -------
    ODESolution
        Solution with x, y arrays and metadata
    """
    # Implementation
    ...
    return ODESolution(x, y, n_evals, True, "converged")
```

2. **Add stability analysis** if applicable

3. **Write comprehensive tests** including:
   - Known analytical solutions
   - Convergence rate verification
   - Edge cases (stiff problems, discontinuities)

### Code Style

- **PEP 8** compliance
- **Type hints** for all public functions
- **Docstrings** in NumPy format
- **Comments** for complex algorithms
- **Meaningful variable names**

Example:
```python
def trapezoidal(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100
) -> IntegrationResult:
    """
    Compute integral using trapezoidal rule.
    
    The trapezoidal rule approximates the integral by summing
    areas of trapezoids under the curve.
    
    Parameters
    ----------
    f : callable
        Function to integrate, must accept float and return float
    a : float
        Lower integration bound
    b : float  
        Upper integration bound
    n : int, optional
        Number of intervals (default: 100)
        
    Returns
    -------
    IntegrationResult
        Named tuple with:
        - value: float - integral approximation
        - error: float - error estimate  
        - n_evaluations: int - function calls made
        - method: str - method name
        
    Raises
    ------
    IntegrationError
        If integration fails or parameters invalid
        
    Examples
    --------
    >>> result = trapezoidal(lambda x: x**2, 0, 1, n=100)
    >>> abs(result.value - 1/3) < 1e-4
    True
    
    Notes
    -----
    Converges as O(h²) where h = (b-a)/n.
    
    References
    ----------
    .. [1] Burden & Faires, "Numerical Analysis", 9th ed.
    """
    # Implementation...
```

### Testing Requirements

1. **Test coverage:** Aim for >90% for new code
2. **Test types:**
   - Unit tests for individual functions
   - Integration tests for workflows
   - Edge cases and error handling
   - Numerical accuracy verification

3. **Run tests:**
```bash
# All tests
pytest

# With coverage
pytest --cov=numeric_integrator --cov-report=term-missing

# Specific file
pytest tests/test_integrators.py -v

# Specific test
pytest tests/test_integrators.py::TestTrapezoidal::test_polynomial -v
```

### Documentation

1. **Docstrings:** All public functions must have docstrings
2. **Examples:** Include usage examples in docstrings
3. **README updates:** Update README.md for new features
4. **Notebooks:** Add notebook examples for major features

### Mathematical Background

When adding new methods, include:

1. **Algorithm description**
2. **Convergence order** (e.g., O(h⁴))
3. **Stability properties** (for ODE methods)
4. **Error estimates**
5. **References** to literature

Example in documentation:
```markdown
#### Your Method

Mathematical formulation:
$$\int_a^b f(x)dx \approx \sum_{i=0}^{n} w_i f(x_i)$$

where weights $w_i$ are derived from...

**Convergence:** O(h^p) where p = ...

**Error estimate:** 
$$E \approx \frac{(b-a)^{p+1}}{(p+1)!} f^{(p)}(\xi)$$

**Best for:** Smooth functions, moderate accuracy requirements

**Avoid for:** Highly oscillatory functions, singularities
```

## Pull Request Process

1. **Fork** the repository
2. **Create branch:** `git checkout -b feature/your-feature-name`
3. **Make changes** following guidelines above
4. **Add tests** ensuring they pass
5. **Update documentation** (README, docstrings)
6. **Commit:** Use clear commit messages
   ```
   Add Gaussian quadrature integration method
   
   - Implement Gauss-Legendre quadrature
   - Add tests for polynomials up to degree 2n-1
   - Update API and README
   - Include convergence analysis
   ```
7. **Push:** `git push origin feature/your-feature-name`
8. **Create Pull Request** with description:
   - What: Brief description of changes
   - Why: Motivation and use cases
   - How: Implementation approach
   - Tests: Coverage and validation
   - Docs: Documentation updates

## Code Review Checklist

Before submitting, verify:

- [ ] Code follows PEP 8 style
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Test coverage >90% for new code
- [ ] Docstrings complete and correct
- [ ] Type hints included
- [ ] README updated if needed
- [ ] No breaking changes (or documented)
- [ ] Performance acceptable (compare benchmarks)
- [ ] No new dependencies (or justified)

## Reporting Issues

When reporting bugs:

1. **Search** existing issues first
2. **Minimal example** reproducing the bug
3. **Environment:** Python version, OS, package versions
4. **Expected vs actual** behavior
5. **Error messages** and tracebacks

Template:
```markdown
## Bug Description
Brief description of the issue

## To Reproduce
```python
from numeric_integrator import integrate
result = integrate(lambda x: 1/x, 0, 1)  # Division by zero at x=0
```

## Expected Behavior
Should raise IntegrationError about singularity

## Actual Behavior
Returns NaN without warning

## Environment
- Python: 3.11.5
- OS: macOS 14.0
- numpy: 1.24.0
- numeric_integrator: 0.1.0

## Additional Context
This happens with any function that has singularities at boundaries
```

## Feature Requests

For new features:

1. **Use case:** Why is this needed?
2. **Proposed solution:** How should it work?
3. **Alternatives:** Other approaches considered?
4. **Examples:** Show usage examples

## Performance Contributions

When optimizing:

1. **Benchmark first:** Establish baseline
2. **Profile:** Identify bottlenecks
3. **Measure improvement:** Quantify speedup
4. **Document tradeoffs:** Accuracy vs speed

## Questions?

- Open an issue for questions
- Check existing documentation first
- Be specific about what you're trying to do

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other contributors

Thank you for contributing to Numeric Integrator! 🎉
