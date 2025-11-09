"""
Example: Computing curvature quantities for a simple metric.

This demonstrates how to:
1. Define a metric tensor
2. Compute Christoffel symbols (simplified)
3. Work with index gymnastics
4. Prepare for curvature computations
"""

import numpy as np
from tas import Tensor
from tas.core.metrics import Metric, euclidean_metric
from tas.core.connections import Connection
from tas.core.algebra import symmetrize, trace
from tas.core.einsum_parser import einsum_eval


def example_flat_space():
    """Example with flat Euclidean space."""
    print("=" * 60)
    print("Example 1: Flat Euclidean Space")
    print("=" * 60)
    
    # Euclidean metric in 3D: g_ij = δ_ij
    g = euclidean_metric(3)
    
    print("\nMetric tensor g_ij:")
    print(g.data)
    print(f"Signature: {g.signature()}")
    print(f"Determinant: {g.determinant()}")
    
    # Inverse metric g^ij
    g_inv = g.inverse()
    print("\nInverse metric g^ij:")
    print(g_inv.data)
    
    # Verify g_ik g^kj = δ^j_i
    # Using direct contraction instead of string parser for now
    from tas.core.einsum_parser import contract
    product = contract(g, g_inv, axes1=(1,), axes2=(0,))
    print("\nProduct g_ik g^kj (should be identity):")
    print(product.data)
    
    # For flat space, Christoffel symbols are zero
    conn = Connection.from_metric(g)
    print("\nChristoffel symbols Γ^k_ij (all zero for flat space):")
    print(f"Max absolute value: {np.max(np.abs(conn.christoffel.data))}")
    print(f"Is torsion-free? {conn.is_symmetric()}")


def example_2d_polar_metric():
    """Example with 2D polar coordinates."""
    print("\n" + "=" * 60)
    print("Example 2: 2D Polar Coordinates")
    print("=" * 60)
    
    # Metric in polar coordinates: ds² = dr² + r² dθ²
    r = 2.0  # Evaluate at r = 2
    
    g_data = np.array([
        [1.0, 0.0],      # g_rr, g_rθ
        [0.0, r**2]      # g_θr, g_θθ
    ])
    
    g = Metric(g_data, indices=("_i", "_j"), name="polar_metric")
    
    print(f"\nPolar metric at r={r}:")
    print(g.data)
    print(f"Determinant: {g.determinant()}")
    print(f"sqrt(det g) = {g.sqrt_abs_det()} (volume element)")
    
    # Inverse metric
    g_inv = g.inverse()
    print("\nInverse metric g^ij:")
    print(g_inv.data)
    
    # Vector in polar coordinates
    v = Tensor(np.array([1.0, 0.5]), indices=("^i",), name="v")
    print(f"\nContravariant vector v^i = {v.data}")
    
    # Lower the index
    v_down = g.lower_index(v, axis=0)
    print(f"Covariant vector v_i = {v_down.data}")
    
    # Compute norm: |v|² = g_ij v^i v^j
    norm_squared = g.inner_product(v, v)
    print(f"\nNorm squared |v|² = {norm_squared:.4f}")
    print(f"Norm |v| = {np.sqrt(norm_squared):.4f}")


def example_minkowski_spacetime():
    """Example with Minkowski spacetime."""
    print("\n" + "=" * 60)
    print("Example 3: Minkowski Spacetime")
    print("=" * 60)
    
    from tas.core.metrics import minkowski_metric
    
    # Minkowski metric with signature (- + + +)
    eta = minkowski_metric("timelike")
    
    print("\nMinkowski metric η_μν:")
    print(eta.data)
    print(f"Signature: {eta.signature()}")
    
    # Four-velocity of a particle moving in x-direction
    # For v = 0.6c, γ = 1/√(1-v²) = 1.25
    gamma = 1.25
    u = Tensor(
        np.array([gamma, gamma * 0.6, 0, 0]),
        indices=("^mu",),
        name="four_velocity"
    )
    
    print(f"\nFour-velocity u^μ = {u.data}")
    
    # Verify normalization: η_μν u^μ u^ν = -1 (timelike)
    norm = eta.inner_product(u, u)
    print(f"Norm η_μν u^μ u^ν = {norm:.6f} (should be -1)")
    
    # Energy-momentum tensor for dust: T^μν = ρ u^μ u^ν
    rho = 1.0  # Rest mass density
    
    from tas.core.einsum_parser import outer
    T = outer(u, u)
    T_scaled = Tensor(
        rho * T.data,
        indices=("^mu", "^nu"),
        name="stress_energy"
    )
    
    print("\nStress-energy tensor T^μν:")
    print(T_scaled.data)
    
    # Trace: T = η_μν T^μν
    T_down = eta.lower_index(T_scaled, axis=0)
    T_down = eta.lower_index(T_down, axis=1)
    T_trace = trace(T_down, axis1=0, axis2=1)
    print(f"\nTrace T = {T_trace.data:.6f}")


def example_stress_energy_conservation():
    """Example showing stress-energy tensor conservation."""
    print("\n" + "=" * 60)
    print("Example 4: Tensor Contractions and Conservation")
    print("=" * 60)
    
    # In GR, stress-energy conservation: ∇_μ T^μν = 0
    # In flat space (no gravity), this becomes: ∂_μ T^μν = 0
    
    # For this simplified example, we'll just show the contraction pattern
    from tas.core.metrics import minkowski_metric
    
    eta = minkowski_metric("timelike")
    
    # Simplified stress tensor (perfect fluid at rest)
    # T^μν = diag(ρ, p, p, p) where ρ=energy density, p=pressure
    rho, p = 1.0, 0.3
    
    T_data = np.diag([rho, p, p, p])
    T = Tensor(T_data, indices=("^mu", "^nu"), name="T")
    
    print("\nStress-energy tensor (perfect fluid at rest):")
    print(T.data)
    
    # Lower both indices: T_μν = η_μα η_νβ T^αβ
    T_down = eta.lower_index(T, axis=0)
    T_down = eta.lower_index(T_down, axis=1)
    
    print("\nFully covariant T_μν:")
    print(T_down.data)
    
    # Mixed form: T^μ_ν (one up, one down)
    T_mixed = eta.lower_index(T, axis=1)
    
    print("\nMixed tensor T^μ_ν:")
    print(T_mixed.data)
    
    # Trace: T = T^μ_μ
    T_trace = trace(T_mixed, axis1=0, axis2=1)
    print(f"\nTrace T^μ_μ = {T_trace.data:.4f}")
    print(f"(For perfect fluid: T = ρ - 3p = {rho - 3*p:.4f})")


if __name__ == "__main__":
    example_flat_space()
    example_2d_polar_metric()
    example_minkowski_spacetime()
    example_stress_energy_conservation()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
