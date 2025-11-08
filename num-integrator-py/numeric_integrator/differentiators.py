"""
numerical derivative approximations using finite difference schemes

implements forward, backward, central differences and richardson extrapolation.
"""

from typing import Callable
import numpy as np
from dataclasses import dataclass

from numeric_integrator.errors import DifferentiationError


@dataclass
class DerivativeResult:
    """result of numerical differentiation with error estimation"""
    value: float
    error: float
    n_evaluations: int
    method: str


def forward_difference(f: Callable[[float], float], x0: float, h: float = 1e-5) -> DerivativeResult:
    """
    forward finite difference approximation
    
    computes: f'(x₀) ≈ [f(x₀ + h) - f(x₀)] / h
    
    accuracy: O(h) first-order accurate
    
    args:
        f: function to differentiate
        x0: point at which to compute derivative
        h: step size (must be positive and small)
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if h is invalid or function evaluation fails
    """
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    if not np.isfinite(x0):
        raise DifferentiationError(f"x0 must be finite, got {x0}")
    
    try:
        f0 = f(x0)
        f1 = f(x0 + h)
    except Exception as e:
        raise DifferentiationError(f"function evaluation failed: {e}")
    
    if not np.isfinite(f0) or not np.isfinite(f1):
        raise DifferentiationError("function returned non-finite values")
    
    derivative = (f1 - f0) / h
    
    # error estimate using second derivative approximation
    try:
        f2 = f(x0 + 2*h)
        second_deriv_approx = abs((f2 - 2*f1 + f0) / (h*h))
        error = 0.5 * h * second_deriv_approx  # truncation error
    except:
        error = abs(derivative) * 1e-3  # rough estimate
    
    return DerivativeResult(
        value=float(derivative),
        error=float(error),
        n_evaluations=2,
        method="forward_difference"
    )


def backward_difference(f: Callable[[float], float], x0: float, h: float = 1e-5) -> DerivativeResult:
    """
    backward finite difference approximation
    
    computes: f'(x₀) ≈ [f(x₀) - f(x₀ - h)] / h
    
    accuracy: O(h) first-order accurate
    
    args:
        f: function to differentiate
        x0: point at which to compute derivative
        h: step size (must be positive and small)
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if h is invalid or function evaluation fails
    """
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    if not np.isfinite(x0):
        raise DifferentiationError(f"x0 must be finite, got {x0}")
    
    try:
        f0 = f(x0)
        f_minus = f(x0 - h)
    except Exception as e:
        raise DifferentiationError(f"function evaluation failed: {e}")
    
    if not np.isfinite(f0) or not np.isfinite(f_minus):
        raise DifferentiationError("function returned non-finite values")
    
    derivative = (f0 - f_minus) / h
    
    # error estimate
    try:
        f_minus2 = f(x0 - 2*h)
        second_deriv_approx = abs((f0 - 2*f_minus + f_minus2) / (h*h))
        error = 0.5 * h * second_deriv_approx
    except:
        error = abs(derivative) * 1e-3
    
    return DerivativeResult(
        value=float(derivative),
        error=float(error),
        n_evaluations=2,
        method="backward_difference"
    )


def central_difference(f: Callable[[float], float], x0: float, h: float = 1e-5) -> DerivativeResult:
    """
    central finite difference approximation
    
    computes: f'(x₀) ≈ [f(x₀ + h) - f(x₀ - h)] / (2h)
    
    accuracy: O(h²) second-order accurate (more accurate than forward/backward)
    
    args:
        f: function to differentiate
        x0: point at which to compute derivative
        h: step size (must be positive and small)
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if h is invalid or function evaluation fails
    """
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    if not np.isfinite(x0):
        raise DifferentiationError(f"x0 must be finite, got {x0}")
    
    try:
        f_plus = f(x0 + h)
        f_minus = f(x0 - h)
    except Exception as e:
        raise DifferentiationError(f"function evaluation failed: {e}")
    
    if not np.isfinite(f_plus) or not np.isfinite(f_minus):
        raise DifferentiationError("function returned non-finite values")
    
    derivative = (f_plus - f_minus) / (2 * h)
    
    # error estimate using third derivative approximation
    try:
        f_plus2 = f(x0 + 2*h)
        f_minus2 = f(x0 - 2*h)
        third_deriv_approx = abs((f_plus2 - 2*f_plus + 2*f_minus - f_minus2) / (2*h*h*h))
        error = (h*h / 6) * third_deriv_approx  # truncation error O(h²)
    except:
        error = abs(derivative) * 1e-6
    
    return DerivativeResult(
        value=float(derivative),
        error=float(error),
        n_evaluations=2,
        method="central_difference"
    )


def richardson_extrapolation(f: Callable[[float], float], x0: float, h: float = 1e-3,
                             n_iter: int = 4) -> DerivativeResult:
    """
    richardson extrapolation for higher accuracy derivatives
    
    uses repeated halving of step size and extrapolation to cancel
    leading error terms. achieves very high accuracy O(h^(2n)).
    
    args:
        f: function to differentiate
        x0: point at which to compute derivative
        h: initial step size
        n_iter: number of extrapolation iterations (higher = more accurate)
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if parameters are invalid or computation fails
    """
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    if not np.isfinite(x0):
        raise DifferentiationError(f"x0 must be finite, got {x0}")
    if n_iter <= 0:
        raise DifferentiationError(f"n_iter must be positive, got {n_iter}")
    
    # richardson tableau
    D = np.zeros((n_iter, n_iter))
    n_evals = 0
    
    # compute initial estimates with decreasing step sizes
    for i in range(n_iter):
        h_i = h / (2**i)
        try:
            f_plus = f(x0 + h_i)
            f_minus = f(x0 - h_i)
            n_evals += 2
        except Exception as e:
            raise DifferentiationError(f"function evaluation failed: {e}")
        
        if not np.isfinite(f_plus) or not np.isfinite(f_minus):
            raise DifferentiationError("function returned non-finite values")
        
        D[i, 0] = (f_plus - f_minus) / (2 * h_i)
    
    # richardson extrapolation
    for j in range(1, n_iter):
        for i in range(j, n_iter):
            D[i, j] = D[i, j-1] + (D[i, j-1] - D[i-1, j-1]) / (4**j - 1)
    
    # best estimate is in bottom-right corner
    value = D[n_iter-1, n_iter-1]
    
    # error estimate from last two diagonal elements
    if n_iter > 1:
        error = abs(D[n_iter-1, n_iter-1] - D[n_iter-2, n_iter-2])
    else:
        error = abs(value) * 1e-10
    
    return DerivativeResult(
        value=float(value),
        error=float(error),
        n_evaluations=n_evals,
        method="richardson_extrapolation"
    )


def derivative_vector(f: Callable[[np.ndarray], float], x: np.ndarray, 
                      h: float = 1e-5) -> np.ndarray:
    """
    compute gradient vector using central differences
    
    computes: ∇f(x) ≈ [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    args:
        f: scalar function of vector input
        x: point at which to compute gradient
        h: step size
    
    returns:
        gradient vector as numpy array
    
    raises:
        DifferentiationError: if computation fails
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    
    n = len(x)
    gradient = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        try:
            f_plus = f(x_plus)
            f_minus = f(x_minus)
        except Exception as e:
            raise DifferentiationError(f"function evaluation failed: {e}")
        
        gradient[i] = (f_plus - f_minus) / (2 * h)
    
    return gradient


def second_derivative(f: Callable[[float], float], x0: float, h: float = 1e-4) -> DerivativeResult:
    """
    second derivative using central difference
    
    computes: f''(x₀) ≈ [f(x₀ + h) - 2f(x₀) + f(x₀ - h)] / h²
    
    accuracy: O(h²)
    
    args:
        f: function to differentiate
        x0: point at which to compute second derivative
        h: step size
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if computation fails
    """
    if h <= 0:
        raise DifferentiationError(f"step size must be positive, got h={h}")
    if not np.isfinite(x0):
        raise DifferentiationError(f"x0 must be finite, got {x0}")
    
    try:
        f0 = f(x0)
        f_plus = f(x0 + h)
        f_minus = f(x0 - h)
    except Exception as e:
        raise DifferentiationError(f"function evaluation failed: {e}")
    
    if not np.isfinite(f0) or not np.isfinite(f_plus) or not np.isfinite(f_minus):
        raise DifferentiationError("function returned non-finite values")
    
    second_deriv = (f_plus - 2*f0 + f_minus) / (h*h)
    
    # error estimate
    try:
        f_plus2 = f(x0 + 2*h)
        f_minus2 = f(x0 - 2*h)
        fourth_deriv_approx = abs((f_plus2 - 4*f_plus + 6*f0 - 4*f_minus + f_minus2) / (h**4))
        error = (h*h / 12) * fourth_deriv_approx
    except:
        error = abs(second_deriv) * 1e-4
    
    return DerivativeResult(
        value=float(second_deriv),
        error=float(error),
        n_evaluations=3,
        method="second_derivative"
    )
