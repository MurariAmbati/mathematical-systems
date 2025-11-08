"""
utility functions for numerical methods

provides interpolation, meshing, and helper functions.
"""

from typing import Callable, List, Tuple, Optional
import numpy as np


def linspace_adaptive(f: Callable[[float], float], a: float, b: float,
                     tol: float = 0.01, min_points: int = 10,
                     max_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    create adaptive mesh based on function curvature
    
    places more points where function varies rapidly.
    
    args:
        f: function to sample
        a: lower bound
        b: upper bound
        tol: tolerance for curvature detection
        min_points: minimum number of points
        max_points: maximum number of points
    
    returns:
        tuple of (x_points, y_points)
    """
    x_points = [a]
    y_points = [f(a)]
    
    def needs_refinement(x1: float, x2: float, x3: float) -> bool:
        """check if interval needs refinement based on curvature"""
        y1, y2, y3 = f(x1), f(x2), f(x3)
        # check if midpoint deviates from linear interpolation
        y_linear = (y1 + y3) / 2
        deviation = abs(y2 - y_linear) / (abs(y_linear) + 1e-10)
        return deviation > tol
    
    # initial uniform mesh
    x = np.linspace(a, b, min_points)
    for xi in x[1:]:
        x_points.append(xi)
        y_points.append(f(xi))
    
    # adaptive refinement
    i = 0
    while i < len(x_points) - 2 and len(x_points) < max_points:
        x1, x2, x3 = x_points[i], x_points[i+1], x_points[i+2]
        
        if needs_refinement(x1, x2, x3):
            # insert midpoints
            x_mid1 = (x1 + x2) / 2
            x_mid2 = (x2 + x3) / 2
            x_points.insert(i+1, x_mid1)
            y_points.insert(i+1, f(x_mid1))
            x_points.insert(i+3, x_mid2)
            y_points.insert(i+3, f(x_mid2))
        else:
            i += 1
    
    return np.array(x_points), np.array(y_points)


def lagrange_interpolation(x_data: np.ndarray, y_data: np.ndarray) -> Callable[[float], float]:
    """
    create lagrange interpolating polynomial
    
    returns callable that evaluates P(x) = Σ yᵢ·Lᵢ(x)
    where Lᵢ(x) = Π(x - xⱼ)/(xᵢ - xⱼ) for j ≠ i
    
    args:
        x_data: x coordinates of data points
        y_data: y coordinates of data points
    
    returns:
        interpolating function
    """
    n = len(x_data)
    
    def interpolant(x: float) -> float:
        result = 0.0
        for i in range(n):
            # compute lagrange basis polynomial Li(x)
            Li = 1.0
            for j in range(n):
                if i != j:
                    Li *= (x - x_data[j]) / (x_data[i] - x_data[j])
            result += y_data[i] * Li
        return result
    
    return interpolant


def newton_divided_differences(x_data: np.ndarray, y_data: np.ndarray) -> Callable[[float], float]:
    """
    create newton interpolating polynomial using divided differences
    
    more numerically stable than lagrange form.
    
    args:
        x_data: x coordinates of data points
        y_data: y coordinates of data points
    
    returns:
        interpolating function
    """
    n = len(x_data)
    
    # compute divided difference table
    coeff = np.zeros((n, n))
    coeff[:, 0] = y_data
    
    for j in range(1, n):
        for i in range(n - j):
            coeff[i, j] = (coeff[i+1, j-1] - coeff[i, j-1]) / (x_data[i+j] - x_data[i])
    
    # extract coefficients from first row
    a = coeff[0, :]
    
    def interpolant(x: float) -> float:
        result = a[0]
        product = 1.0
        for i in range(1, n):
            product *= (x - x_data[i-1])
            result += a[i] * product
        return result
    
    return interpolant


def cubic_spline(x_data: np.ndarray, y_data: np.ndarray, 
                bc_type: str = "natural") -> Callable[[float], float]:
    """
    create cubic spline interpolant
    
    piecewise cubic polynomials with continuous first and second derivatives.
    
    args:
        x_data: x coordinates (must be sorted)
        y_data: y coordinates
        bc_type: boundary condition type ("natural", "clamped", or "periodic")
    
    returns:
        spline interpolating function
    """
    n = len(x_data) - 1
    h = np.diff(x_data)
    
    # build tridiagonal system for second derivatives
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    
    if bc_type == "natural":
        # natural spline: S''(x₀) = S''(xₙ) = 0
        A[0, 0] = 1
        A[n, n] = 1
        start_row = 1
    elif bc_type == "clamped":
        # clamped spline: S'(x₀) = f'(x₀), S'(xₙ) = f'(xₙ)
        # approximate derivatives with finite differences
        A[0, 0] = 2*h[0]
        A[0, 1] = h[0]
        b[0] = 3 * (y_data[1] - y_data[0])
        A[n, n-1] = h[n-1]
        A[n, n] = 2*h[n-1]
        b[n] = 3 * (y_data[n] - y_data[n-1])
        start_row = 1
    else:
        start_row = 0
    
    # interior equations
    for i in range(start_row, n):
        if i > 0:
            A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i]) if i < n else 2*h[i-1]
        if i < n:
            A[i, i+1] = h[i]
            b[i] = 3 * ((y_data[i+1] - y_data[i])/h[i] - (y_data[i] - y_data[i-1])/h[i-1])
    
    # solve for second derivatives
    try:
        M = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # fallback to linear interpolation
        return lambda x: np.interp(x, x_data, y_data)
    
    def spline_eval(x: float) -> float:
        # find interval containing x
        if x <= x_data[0]:
            return y_data[0]
        if x >= x_data[-1]:
            return y_data[-1]
        
        i = np.searchsorted(x_data, x) - 1
        i = min(max(i, 0), n-1)
        
        # evaluate cubic polynomial on interval [xi, xi+1]
        hi = h[i]
        t = (x - x_data[i]) / hi
        
        a = y_data[i]
        b_coeff = (y_data[i+1] - y_data[i])/hi - hi*(2*M[i] + M[i+1])/3
        c = M[i]
        d = (M[i+1] - M[i]) / (3*hi)
        
        return a + b_coeff*t*hi + c*(t*hi)**2 + d*(t*hi)**3
    
    return spline_eval


def chebyshev_nodes(n: int, a: float = -1, b: float = 1) -> np.ndarray:
    """
    generate chebyshev nodes for polynomial interpolation
    
    minimizes runge phenomenon by clustering points near endpoints.
    
    nodes: xᵢ = cos((2i + 1)π / (2n))
    
    args:
        n: number of nodes
        a: lower bound
        b: upper bound
    
    returns:
        array of chebyshev nodes
    """
    i = np.arange(n)
    nodes = np.cos((2*i + 1) * np.pi / (2*n))
    
    # map from [-1, 1] to [a, b]
    nodes = (b - a) / 2 * nodes + (b + a) / 2
    
    return nodes


def adaptive_mesh_refinement(x: np.ndarray, y: np.ndarray,
                            tolerance: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    refine mesh where solution varies rapidly
    
    inserts points where adjacent segments have large curvature.
    
    args:
        x: current x mesh
        y: current y values
        tolerance: curvature threshold
    
    returns:
        refined (x, y) mesh
    """
    x_refined = [x[0]]
    y_refined = [y[0]]
    
    for i in range(len(x) - 1):
        x_refined.append(x[i+1])
        y_refined.append(y[i+1])
        
        # check if refinement needed
        if i < len(x) - 2:
            # estimate second derivative
            dx1 = x[i+1] - x[i]
            dx2 = x[i+2] - x[i+1]
            if dx1 > 0 and dx2 > 0:
                d2y = abs((y[i+2] - y[i+1])/dx2 - (y[i+1] - y[i])/dx1) / ((dx1 + dx2)/2)
                
                if d2y > tolerance:
                    # insert midpoint
                    x_mid = (x[i] + x[i+1]) / 2
                    y_mid = (y[i] + y[i+1]) / 2  # linear approximation
                    x_refined.insert(-1, x_mid)
                    y_refined.insert(-1, y_mid)
    
    return np.array(x_refined), np.array(y_refined)


def function_smoother(y: np.ndarray, window: int = 3) -> np.ndarray:
    """
    smooth noisy function values using moving average
    
    args:
        y: array of function values
        window: window size for averaging (must be odd)
    
    returns:
        smoothed array
    """
    if window % 2 == 0:
        window += 1
    
    pad = window // 2
    y_padded = np.pad(y, pad, mode='edge')
    
    smoothed = np.convolve(y_padded, np.ones(window)/window, mode='valid')
    
    return smoothed


def estimate_derivative_order(h_values: List[float], errors: List[float]) -> float:
    """
    estimate order of accuracy from error vs step size data
    
    fits log(error) = c + p*log(h) and returns p.
    
    args:
        h_values: list of step sizes
        errors: corresponding errors
    
    returns:
        estimated order p
    """
    if len(h_values) < 2:
        return 0.0
    
    log_h = np.log(h_values)
    log_err = np.log(np.maximum(errors, 1e-16))  # avoid log(0)
    
    # linear regression
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    p, c = np.linalg.lstsq(A, log_err, rcond=None)[0]
    
    return float(p)


def relative_difference(a: float, b: float, tol: float = 1e-10) -> float:
    """
    compute relative difference between two values
    
    handles case when values are near zero.
    
    args:
        a: first value
        b: second value
        tol: tolerance for zero check
    
    returns:
        relative difference
    """
    max_val = max(abs(a), abs(b))
    if max_val < tol:
        return abs(a - b)
    return abs(a - b) / max_val


def is_converged(current: float, previous: float, tol: float) -> bool:
    """
    check convergence based on relative change
    
    args:
        current: current iteration value
        previous: previous iteration value
        tol: convergence tolerance
    
    returns:
        True if converged
    """
    return relative_difference(current, previous) < tol


def optimal_step_size(f: Callable[[float], float], x: float, 
                     order: int = 2, epsilon: float = 1e-16) -> float:
    """
    compute optimal step size for finite differences
    
    balances truncation error and roundoff error.
    optimal h ≈ (ε)^(1/(order+1)) where ε is machine precision.
    
    args:
        f: function to differentiate
        x: point of interest
        order: order of derivative
        epsilon: machine precision
    
    returns:
        optimal step size
    """
    h_opt = epsilon ** (1.0 / (order + 1))
    
    # scale by function magnitude
    try:
        fx = abs(f(x))
        if fx > 1:
            h_opt *= fx ** (1.0 / (order + 1))
    except:
        pass
    
    return h_opt
