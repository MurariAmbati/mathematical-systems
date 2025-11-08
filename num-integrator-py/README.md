# numeric integrator

python library for numerical integration, differentiation, and ode solving

## installation

```bash
git clone https://github.com/yourusername/numeric-integrator.git
cd numeric-integrator
pip install -e .
```

## usage

### integration

```python
from numeric_integrator import integrate
import numpy as np

# integrate sin(x) from 0 to π
result = integrate(np.sin, 0, np.pi, method='simpson', n=100)
print(result.value)  # 2.000000
```

### differentiation

```python
from numeric_integrator import differentiate

# derivative of e^x at x=1
result = differentiate(np.exp, 1.0, method='central')
print(result.value)  # 2.718282
```

### ode solving

```python
from numeric_integrator import solve_ode

# solve dy/dx = y, y(0) = 1
sol = solve_ode(lambda x, y: y, 1.0, 0, 2, method='rk4', step=0.1)
print(sol.y[-1])  # 7.389056 (≈ e^2)
```

## benchmark results

### integration (n=1000 points)

| method | time (μs) | speedup vs scipy | error (e^x, 0→1) |
|--------|-----------|------------------|------------------|
| scipy.quad | 47 | 1.00x | 4.4e-16 |
| romberg | 66 | 0.71x | 4.4e-16 |
| adaptive simpson | 122 | 0.39x | 1.8e-08 |
| simpson | 2328 | 0.02x | 9.8e-15 |
| trapezoidal | 2371 | 0.02x | 1.4e-07 |

**key findings:**
- romberg matches scipy accuracy at comparable speed
- adaptive methods 3-20x faster than fixed methods
- all methods achieve high accuracy for smooth functions

### ode solvers (exponential growth, 200 steps)

| method | time (μs) | error at t=2 | evaluations |
|--------|-----------|--------------|-------------|
| scipy.dop853 | 323 | 4.5e-07 | 38 |
| rkf45 (ours) | 295 | 2.4e-06 | 84 |
| scipy.rk45 | 530 | 2.4e-06 | 56 |
| heun | 606 | 2.4e-04 | 400 |
| rk4 | 1083 | 1.2e-09 | 800 |
| scipy.rk23 | 2802 | 2.1e-05 | 191 |

**key findings:**
- rkf45 (adaptive) fastest among our methods
- rk4 achieves best accuracy (1.2e-09 error)
- adaptive methods use fewer evaluations
- competitive with scipy for moderate accuracy needs

## methods available

### integration
- `trapezoidal` - o(h²) convergence
- `simpson` - o(h⁴) convergence
- `midpoint` - o(h²) convergence
- `boole` - o(h⁶) convergence
- `romberg` - richardson extrapolation
- `adaptive_trapezoidal` - automatic refinement
- `adaptive_simpson` - automatic refinement

### differentiation
- `forward` - o(h) accuracy
- `backward` - o(h) accuracy
- `central` - o(h²) accuracy
- `richardson` - iterative refinement

### ode solvers
- `euler` - o(h) per step
- `heun` - o(h²) per step
- `rk4` - o(h⁴) per step
- `rkf45` - adaptive o(h⁵)
- `ab2`, `ab3`, `ab4` - adams-bashforth
- `am2`, `am3`, `am4` - adams-moulton

## running benchmarks

```bash
cd examples
python benchmark.py
```

generates:
- console output with timing and accuracy
- `convergence_study.png` - error vs step size
- `performance_scaling.png` - time vs problem size

## examples

see `examples/` directory:
- `integration_demo.ipynb` - integration methods
- `differentiation_demo.ipynb` - derivative computation
- `ode_demo.ipynb` - van der pol oscillator
- `benchmark.py` - scipy comparison

## testing

```bash
pytest                                    # run all tests
pytest --cov=numeric_integrator          # with coverage
```

**test results:**
- 203 tests total
- 186 passing (91.6%)
- 83.4% code coverage

## performance summary

**integration:**
- romberg: best for smooth functions (matches scipy accuracy)
- adaptive simpson: good balance of speed and accuracy
- fixed methods: predictable, simple, slower for large n

**ode solving:**
- rkf45: best general purpose (adaptive, efficient)
- rk4: best accuracy for non-stiff problems
- euler/heun: educational, simple, less accurate

**when to use:**
- high accuracy needed: romberg, rk4
- efficiency important: adaptive methods (rkf45, adaptive simpson)
- simplicity: trapezoidal, euler
- stiff problems: use scipy (implicit methods)

## dependencies

- python 3.9+
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.11.0 (for benchmarking)

## license

mit license - see LICENSE file

---

*pure python implementation for educational and research purposes*
