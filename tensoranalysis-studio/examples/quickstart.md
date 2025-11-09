# Tensor Analysis Studio - Quick Start

This notebook demonstrates the core features of the Tensor Analysis Studio (TAS) library.

## Installation

```bash
pip install tensor-analysis-studio
```

## 1. Basic Tensor Operations

```python
import numpy as np
from tas import Tensor, Index
from tas.core.metrics import euclidean_metric, minkowski_metric
from tas.core.einsum_parser import einsum_eval

# Create a rank-2 tensor (matrix) with explicit indices
A = Tensor(
    data=np.array([[1, 2], [3, 4]]),
    indices=("^i", "_j"),
    name="A"
)

print("Tensor A:")
print(A)
print(f"Shape: {A.shape}, Rank: {A.rank}")
```

## 2. Einstein Notation and Contractions

```python
# Matrix multiplication using Einstein notation
B = Tensor(
    data=np.array([[5, 6], [7, 8]]),
    indices=("^j", "_k"),
    name="B"
)

# Contract: C^i_k = A^i_j B^j_k
C = einsum_eval("A^i_j B^j_k", A=A, B=B)

print("\nMatrix product C = A @ B:")
print(C.data)
print(f"Result indices: {[str(idx) for idx in C.indices]}")
```

## 3. Vectors and Dot Products

```python
# Create vectors with different variance
v = Tensor(np.array([1, 2, 3]), indices=("^i",), name="v")
w = Tensor(np.array([4, 5, 6]), indices=("_i",), name="w")

# Dot product: v^i w_i
dot_product = einsum_eval("v^i w_i", v=v, w=w)

print(f"\nDot product v·w = {dot_product.data}")
```

## 4. Metric Tensors

```python
# Euclidean metric in 3D
g = euclidean_metric(3)

print("\nEuclidean metric:")
print(g.data)
print(f"Signature: {g.signature()}")  # (3, 0, 0) = all positive

# Minkowski spacetime metric
eta = minkowski_metric("timelike")

print("\nMinkowski metric (- + + + signature):")
print(eta.data)
print(f"Signature: {eta.signature()}")  # (3, 1, 0)
```

## 5. Raising and Lowering Indices

```python
# Covariant vector
V_down = Tensor(np.array([1, 2, 3]), indices=("_i",), name="V")

# Raise index: V^i = g^ij V_j
V_up = g.raise_index(V_down, axis=0)

print("\nOriginal covariant vector:")
print(V_down.data, V_down.indices[0])

print("\nRaised to contravariant:")
print(V_up.data, V_up.indices[0])

# Lower it back
V_back = g.lower_index(V_up, axis=0)
print("\nLowered back (should match original):")
print(V_back.data, V_back.indices[0])
```

## 6. Inner Products with Metrics

```python
# Inner product using metric
u = Tensor(np.array([1, 0, 0]), indices=("^i",), name="u")
v = Tensor(np.array([0, 1, 0]), indices=("^j",), name="v")

inner = g.inner_product(u, v)
print(f"\nInner product <u, v> = {inner}")  # Should be 0 (orthogonal)

# Self inner product
self_inner = g.inner_product(u, u)
print(f"Inner product <u, u> = {self_inner}")  # Should be 1
```

## 7. Christoffel Symbols and Connections

```python
from tas.core.connections import Connection

# For a flat metric (Euclidean), Christoffel symbols are zero
conn = Connection.from_metric(g)

print("\nChristoffel symbols (all zero for flat space):")
print(conn.christoffel.data)
print(f"Is torsion-free (symmetric)? {conn.is_symmetric()}")
```

## 8. Coordinate Systems

```python
from tas.core.coords import CartesianFrame, SphericalFrame, CylindricalFrame

# Define coordinate frames
cartesian = CartesianFrame(dim=3)
spherical = SphericalFrame()

# Point in Cartesian coordinates
point_cart = np.array([1.0, 1.0, 1.0])

# Convert to spherical
point_sph = spherical.from_cartesian(point_cart)
print(f"\nCartesian {point_cart} -> Spherical {point_sph}")
print(f"  r = {point_sph[0]:.3f}")
print(f"  θ = {point_sph[1]:.3f} rad = {np.degrees(point_sph[1]):.1f}°")
print(f"  φ = {point_sph[2]:.3f} rad = {np.degrees(point_sph[2]):.1f}°")

# Get Jacobian
J = spherical.jacobian(point_sph)
print(f"\nJacobian matrix at this point:")
print(J)

# Metric in spherical coordinates
g_sph = spherical.metric(point_sph)
print(f"\nSpherical metric at r={point_sph[0]:.3f}, θ={point_sph[1]:.3f}:")
print(g_sph.data)
```

## 9. Tensor Algebra Operations

```python
from tas.core.algebra import symmetrize, antisymmetrize, trace

# Create a non-symmetric tensor
T = Tensor(np.random.rand(3, 3), indices=("^i", "^j"), name="T")

# Symmetrize
T_sym = symmetrize(T, axes=(0, 1))
print("\nSymmetrized tensor T_(ij):")
print(T_sym.data)
print(f"Check symmetry: max|T_ij - T_ji| = {np.max(np.abs(T_sym.data - T_sym.data.T))}")

# Antisymmetrize
T_antisym = antisymmetrize(T, axes=(0, 1))
print("\nAntisymmetrized tensor T_[ij]:")
print(T_antisym.data)
print(f"Check antisymmetry: max|T_ij + T_ji| = {np.max(np.abs(T_antisym.data + T_antisym.data.T))}")

# Trace
M = Tensor(np.array([[1, 2], [3, 4]]), indices=("^i", "_i"), name="M")
tr = trace(M, axis1=0, axis2=1)
print(f"\nTrace of matrix: {tr.data}")
```

## 10. General Relativity Example: Schwarzschild Metric

```python
# Schwarzschild metric in (t, r, θ, φ) coordinates
# ds² = -(1 - 2M/r)dt² + (1 - 2M/r)^(-1)dr² + r²dθ² + r²sin²(θ)dφ²

def schwarzschild_metric(r, theta, M=1.0):
    """Schwarzschild metric at given r, θ."""
    f = 1 - 2*M/r
    g_data = np.diag([
        -f,           # g_tt
        1/f,          # g_rr
        r**2,         # g_θθ
        r**2 * np.sin(theta)**2  # g_φφ
    ])
    
    return Metric(g_data, indices=("_mu", "_nu"), name="Schwarzschild")

# Evaluate at r=10M, θ=π/4
g_sch = schwarzschild_metric(r=10.0, theta=np.pi/4, M=1.0)

print("\nSchwarzschild metric at r=10M, θ=π/4:")
print(g_sch.data)
print(f"Signature: {g_sch.signature()}")  # (-1, 3, 0) Lorentzian
print(f"Determinant: {g_sch.determinant():.6f}")
```

## Summary

This notebook demonstrated:
- Creating tensors with explicit index labels
- Einstein notation for tensor contractions
- Metric tensors and raising/lowering indices
- Coordinate systems and transformations
- Tensor algebra operations
- A glimpse of general relativity applications

For more examples, see the `examples/` directory in the repository.

## Next Steps

- Explore covariant derivatives with `tas.core.diffops`
- Visualize tensor fields with `tas.viz`
- Try symbolic computations with `tas.sym` (requires sympy)
- Experiment with JAX backend for autodiff and GPU acceleration
