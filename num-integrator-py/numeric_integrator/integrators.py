"""
deterministic quadrature methods for numerical integration

implements classical and modern integration schemes with error estimation.
"""

from typing import Callable, Tuple
import numpy as np
from dataclasses import dataclass

from numeric_integrator.errors import IntegrationError, ConvergenceError


@dataclass
class IntegrationResult:
    """result of numerical integration with error estimation"""
    value: float
    error: float
    n_evaluations: int
    method: str


def trapezoidal(f: Callable[[float], float], a: float, b: float, n: int = 100) -> IntegrationResult:
    """
    trapezoidal rule for numerical integration
    
    computes: ∫[a,b] f(x) dx ≈ h/2 * [f(a) + 2*∑f(xᵢ) + f(b)]
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of subintervals (must be positive)
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if n <= 0 or bounds are invalid
    """
    if n <= 0:
        raise IntegrationError(f"number of intervals must be positive, got {n}")
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x])
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    if not np.all(np.isfinite(y)):
        raise IntegrationError("function returned non-finite values")
    
    # trapezoidal rule
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    # error estimate using Richardson extrapolation
    if n >= 4:
        h2 = (b - a) / (n // 2)
        x2 = np.linspace(a, b, n // 2 + 1)
        y2 = np.array([f(xi) for xi in x2])
        integral2 = h2 * (0.5 * y2[0] + np.sum(y2[1:-1]) + 0.5 * y2[-1])
        error = abs(integral - integral2) / 3.0  # O(h²) error
    else:
        error = abs(integral) * 1e-6  # rough estimate
    
    return IntegrationResult(
        value=float(integral),
        error=float(error),
        n_evaluations=n + 1,
        method="trapezoidal"
    )


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 100) -> IntegrationResult:
    """
    simpson's rule (composite form) for numerical integration
    
    computes: ∫[a,b] f(x) dx ≈ h/3 * [f(a) + 4*∑f(x₂ᵢ₊₁) + 2*∑f(x₂ᵢ) + f(b)]
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of subintervals (must be even and positive)
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if n is not even and positive, or bounds are invalid
    """
    if n <= 0:
        raise IntegrationError(f"number of intervals must be positive, got {n}")
    if n % 2 != 0:
        raise IntegrationError(f"simpson's rule requires even number of intervals, got {n}")
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x])
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    if not np.all(np.isfinite(y)):
        raise IntegrationError("function returned non-finite values")
    
    # simpson's 1/3 rule
    integral = h / 3.0 * (
        y[0] + 
        4.0 * np.sum(y[1:-1:2]) +  # odd indices
        2.0 * np.sum(y[2:-1:2]) +  # even indices
        y[-1]
    )
    
    # error estimate using Richardson extrapolation
    if n >= 8:
        h2 = (b - a) / (n // 2)
        x2 = np.linspace(a, b, n // 2 + 1)
        y2 = np.array([f(xi) for xi in x2])
        integral2 = h2 / 3.0 * (
            y2[0] + 
            4.0 * np.sum(y2[1:-1:2]) +
            2.0 * np.sum(y2[2:-1:2]) +
            y2[-1]
        )
        error = abs(integral - integral2) / 15.0  # O(h⁴) error
    else:
        error = abs(integral) * 1e-8  # rough estimate
    
    return IntegrationResult(
        value=float(integral),
        error=float(error),
        n_evaluations=n + 1,
        method="simpson"
    )


def midpoint(f: Callable[[float], float], a: float, b: float, n: int = 100) -> IntegrationResult:
    """
    midpoint rule for numerical integration
    
    computes: ∫[a,b] f(x) dx ≈ h * ∑f((xᵢ + xᵢ₊₁)/2)
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of subintervals (must be positive)
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if n <= 0 or bounds are invalid
    """
    if n <= 0:
        raise IntegrationError(f"number of intervals must be positive, got {n}")
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    h = (b - a) / n
    # evaluate at midpoints
    x_mid = np.linspace(a + h/2, b - h/2, n)
    
    try:
        y = np.array([f(xi) for xi in x_mid])
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    if not np.all(np.isfinite(y)):
        raise IntegrationError("function returned non-finite values")
    
    integral = h * np.sum(y)
    
    # error estimate
    if n >= 4:
        h2 = (b - a) / (n // 2)
        x_mid2 = np.linspace(a + h2/2, b - h2/2, n // 2)
        y2 = np.array([f(xi) for xi in x_mid2])
        integral2 = h2 * np.sum(y2)
        error = abs(integral - integral2) / 3.0
    else:
        error = abs(integral) * 1e-6
    
    return IntegrationResult(
        value=float(integral),
        error=float(error),
        n_evaluations=n,
        method="midpoint"
    )


def boole(f: Callable[[float], float], a: float, b: float, n: int = 100) -> IntegrationResult:
    """
    boole's rule for numerical integration (fifth-order accuracy)
    
    computes: ∫[a,b] f(x) dx using 5-point Newton-Cotes formula
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        n: number of subintervals (must be divisible by 4)
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if n is not divisible by 4 or bounds are invalid
    """
    if n <= 0:
        raise IntegrationError(f"number of intervals must be positive, got {n}")
    if n % 4 != 0:
        raise IntegrationError(f"boole's rule requires n divisible by 4, got {n}")
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    
    try:
        y = np.array([f(xi) for xi in x])
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    if not np.all(np.isfinite(y)):
        raise IntegrationError("function returned non-finite values")
    
    # boole's rule coefficients: 7, 32, 12, 32, 7 (repeating)
    integral = 0.0
    for i in range(0, n, 4):
        integral += (2 * h / 45) * (
            7 * y[i] + 
            32 * y[i+1] + 
            12 * y[i+2] + 
            32 * y[i+3] + 
            7 * y[i+4]
        )
    
    # error estimate (rough)
    error = abs(integral) * 1e-10  # O(h⁶) error
    
    return IntegrationResult(
        value=float(integral),
        error=float(error),
        n_evaluations=n + 1,
        method="boole"
    )


def romberg(f: Callable[[float], float], a: float, b: float, max_iter: int = 10, 
            tol: float = 1e-10) -> IntegrationResult:
    """
    romberg integration using richardson extrapolation
    
    iteratively refines trapezoidal rule estimates using extrapolation.
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        max_iter: maximum number of iterations
        tol: convergence tolerance
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if bounds are invalid
        ConvergenceError: if method fails to converge within max_iter
    """
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    if max_iter <= 0:
        raise IntegrationError(f"max_iter must be positive, got {max_iter}")
    
    # romberg tableau
    R = np.zeros((max_iter, max_iter))
    n_evals = 0
    
    # first trapezoidal estimate
    h = b - a
    try:
        R[0, 0] = 0.5 * h * (f(a) + f(b))
        n_evals += 2
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    for i in range(1, max_iter):
        # refined trapezoidal rule
        h /= 2
        sum_new = 0.0
        n_points = 2**(i-1)
        for k in range(n_points):
            x = a + (2*k + 1) * h
            try:
                sum_new += f(x)
                n_evals += 1
            except Exception as e:
                raise IntegrationError(f"function evaluation failed: {e}")
        
        R[i, 0] = 0.5 * R[i-1, 0] + h * sum_new
        
        # richardson extrapolation
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
        
        # check convergence
        if i > 0:
            error = abs(R[i, i] - R[i-1, i-1])
            if error < tol:
                return IntegrationResult(
                    value=float(R[i, i]),
                    error=float(error),
                    n_evaluations=n_evals,
                    method="romberg"
                )
    
    # did not converge
    raise ConvergenceError(f"romberg integration did not converge within {max_iter} iterations")


def adaptive_trapezoidal(f: Callable[[float], float], a: float, b: float, 
                         tol: float = 1e-8, max_depth: int = 20) -> IntegrationResult:
    """
    adaptive trapezoidal rule with automatic refinement
    
    recursively subdivides intervals where error estimate exceeds tolerance.
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        tol: error tolerance per interval
        max_depth: maximum recursion depth
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if bounds are invalid or max depth exceeded
    """
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    n_evals = [0]  # mutable counter
    
    def adaptive_helper(a: float, b: float, fa: float, fb: float, 
                       whole: float, depth: int) -> Tuple[float, float]:
        """recursive adaptive integration"""
        if depth >= max_depth:
            raise IntegrationError(f"maximum recursion depth {max_depth} exceeded")
        
        # compute midpoint
        c = (a + b) / 2
        try:
            fc = f(c)
            n_evals[0] += 1
        except Exception as e:
            raise IntegrationError(f"function evaluation failed: {e}")
        
        # left and right trapezoids
        h = (b - a) / 2
        left = h / 2 * (fa + fc)
        right = h / 2 * (fc + fb)
        refined = left + right
        
        # error estimate
        error = abs(refined - whole)
        
        if error < 15 * tol:  # 15 is safety factor for O(h²) error
            return refined, error
        else:
            # subdivide
            left_val, left_err = adaptive_helper(a, c, fa, fc, left, depth + 1)
            right_val, right_err = adaptive_helper(c, b, fc, fb, right, depth + 1)
            return left_val + right_val, left_err + right_err
    
    # initial estimate
    try:
        fa = f(a)
        fb = f(b)
        n_evals[0] += 2
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    whole = (b - a) / 2 * (fa + fb)
    
    value, error = adaptive_helper(a, b, fa, fb, whole, 0)
    
    return IntegrationResult(
        value=float(value),
        error=float(error),
        n_evaluations=n_evals[0],
        method="adaptive_trapezoidal"
    )


def adaptive_simpson(f: Callable[[float], float], a: float, b: float, 
                     tol: float = 1e-10, max_depth: int = 20) -> IntegrationResult:
    """
    adaptive simpson's rule with automatic refinement
    
    recursively subdivides intervals where error estimate exceeds tolerance.
    uses simpson's rule locally with O(h⁴) accuracy.
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        tol: error tolerance per interval
        max_depth: maximum recursion depth
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if bounds are invalid or max depth exceeded
    """
    if not np.isfinite(a) or not np.isfinite(b):
        raise IntegrationError(f"bounds must be finite: a={a}, b={b}")
    if a >= b:
        raise IntegrationError(f"lower bound must be less than upper bound: a={a}, b={b}")
    
    n_evals = [0]  # mutable counter
    
    def simpson_interval(a: float, b: float, fa: float, fb: float, fc: float) -> float:
        """simpson's rule for interval [a, b]"""
        h = (b - a) / 2
        return h / 3 * (fa + 4*fc + fb)
    
    def adaptive_helper(a: float, b: float, fa: float, fb: float, fc: float,
                       whole: float, depth: int) -> Tuple[float, float]:
        """recursive adaptive integration"""
        if depth >= max_depth:
            raise IntegrationError(f"maximum recursion depth {max_depth} exceeded")
        
        # midpoints of left and right halves
        c = (a + b) / 2
        d = (a + c) / 2
        e = (c + b) / 2
        
        try:
            fd = f(d)
            fe = f(e)
            n_evals[0] += 2
        except Exception as e:
            raise IntegrationError(f"function evaluation failed: {e}")
        
        # simpson on left and right halves
        left = simpson_interval(a, c, fa, fc, fd)
        right = simpson_interval(c, b, fc, fb, fe)
        refined = left + right
        
        # error estimate (O(h⁴))
        error = abs(refined - whole) / 15.0
        
        if error < tol:
            return refined + error, error  # add correction term
        else:
            # subdivide
            left_val, left_err = adaptive_helper(a, c, fa, fc, fd, left, depth + 1)
            right_val, right_err = adaptive_helper(c, b, fc, fb, fe, right, depth + 1)
            return left_val + right_val, left_err + right_err
    
    # initial estimate
    try:
        fa = f(a)
        fb = f(b)
        fc = f((a + b) / 2)
        n_evals[0] += 3
    except Exception as e:
        raise IntegrationError(f"function evaluation failed: {e}")
    
    whole = simpson_interval(a, b, fa, fb, fc)
    
    value, error = adaptive_helper(a, b, fa, fb, fc, whole, 0)
    
    return IntegrationResult(
        value=float(value),
        error=float(error),
        n_evaluations=n_evals[0],
        method="adaptive_simpson"
    )
