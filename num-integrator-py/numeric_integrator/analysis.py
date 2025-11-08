"""
error estimation, stability analysis, and convergence diagnostics

provides tools for analyzing numerical method accuracy and performance.
"""

from typing import Callable, List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

from numeric_integrator.errors import ConvergenceError, StabilityError


@dataclass
class ErrorAnalysis:
    """result of error analysis"""
    absolute_error: float
    relative_error: float
    method: str
    parameters: Dict[str, Any]


@dataclass
class ConvergenceResult:
    """result of convergence analysis"""
    converged: bool
    rate: float  # convergence rate (order of accuracy)
    errors: List[float]
    step_sizes: List[float]
    method: str


@dataclass
class StabilityResult:
    """result of stability analysis"""
    stable: bool
    spectral_radius: float
    max_eigenvalue: float
    method: str
    message: str


def estimate_error(computed_value: float, exact_value: float, 
                   method: str = "unknown") -> ErrorAnalysis:
    """
    estimate absolute and relative errors
    
    computes:
    - absolute error: |computed - exact|
    - relative error: |computed - exact| / |exact|
    
    args:
        computed_value: numerically computed result
        exact_value: known exact solution
        method: name of numerical method used
    
    returns:
        ErrorAnalysis with error metrics
    """
    abs_error = abs(computed_value - exact_value)
    
    if abs(exact_value) > 1e-15:
        rel_error = abs_error / abs(exact_value)
    else:
        rel_error = abs_error  # avoid division by zero
    
    return ErrorAnalysis(
        absolute_error=abs_error,
        relative_error=rel_error,
        method=method,
        parameters={"computed": computed_value, "exact": exact_value}
    )


def convergence_test(method: Callable[[float], float], exact_value: float,
                     step_sizes: List[float], method_name: str = "unknown") -> ConvergenceResult:
    """
    test convergence rate of numerical method
    
    runs method with decreasing step sizes and computes convergence rate
    from log-log plot of error vs step size.
    
    expected relationship: error ∝ h^p where p is order of accuracy
    
    args:
        method: callable that takes step size and returns computed value
        exact_value: known exact solution
        step_sizes: list of decreasing step sizes to test
        method_name: name of method being tested
    
    returns:
        ConvergenceResult with convergence rate and error data
    
    raises:
        ConvergenceError: if method fails or produces invalid results
    """
    if len(step_sizes) < 2:
        raise ConvergenceError("need at least 2 step sizes for convergence test")
    
    errors = []
    
    for h in step_sizes:
        try:
            computed = method(h)
        except Exception as e:
            raise ConvergenceError(f"method failed with h={h}: {e}")
        
        if not np.isfinite(computed):
            raise ConvergenceError(f"method returned non-finite value with h={h}")
        
        error = abs(computed - exact_value)
        errors.append(error)
    
    # compute convergence rate from log-log slope
    # log(error) = log(C) + p*log(h)
    # p = Δlog(error) / Δlog(h)
    
    if len(errors) >= 2 and errors[0] > 0 and errors[-1] > 0:
        log_h = np.log(step_sizes)
        log_err = np.log(errors)
        
        # linear regression
        A = np.vstack([log_h, np.ones(len(log_h))]).T
        rate, _ = np.linalg.lstsq(A, log_err, rcond=None)[0]
        
        # check convergence
        converged = errors[-1] < errors[0]
    else:
        rate = 0.0
        converged = False
    
    return ConvergenceResult(
        converged=converged,
        rate=float(rate),
        errors=errors,
        step_sizes=step_sizes,
        method=method_name
    )


def stability_analysis(method: str, step_size: float, lambda_val: complex) -> StabilityResult:
    """
    analyze stability of ode method using test equation dy/dx = λy
    
    for linear stability analysis, we examine the amplification factor
    when applied to dy/dx = λy with solution y = y₀*exp(λx).
    
    method is stable if |amplification factor| ≤ 1 for Re(λ) ≤ 0.
    
    args:
        method: name of ode method ("euler", "heun", "rk4", etc.)
        step_size: integration step size h
        lambda_val: eigenvalue λ (can be complex)
    
    returns:
        StabilityResult with stability information
    
    raises:
        StabilityError: if method is not recognized
    """
    h = step_size
    lam = lambda_val
    z = h * lam  # stability parameter
    
    # stability functions for different methods
    if method.lower() == "euler":
        # yₙ₊₁ = yₙ + h*λ*yₙ = (1 + h*λ)*yₙ
        amplification = 1 + z
    
    elif method.lower() == "heun":
        # yₙ₊₁ = yₙ + h*(k₁ + k₂)/2
        # k₁ = λ*yₙ, k₂ = λ*(yₙ + h*k₁)
        # amplification = 1 + z + z²/2
        amplification = 1 + z + z*z/2
    
    elif method.lower() == "rk4":
        # fourth-order runge-kutta stability function
        amplification = 1 + z + z*z/2 + z*z*z/6 + z*z*z*z/24
    
    elif method.lower() in ["adams_bashforth_2", "ab2"]:
        # yₙ₊₁ = yₙ + h*(3*fₙ - fₙ₋₁)/2
        # characteristic polynomial: r² - r - z*(3*r - 1)/2 = 0
        # for stability, need |r| ≤ 1
        a = 1 - 3*z/2
        b = -1 + z/2
        discriminant = a*a + 4*b
        r1 = (a + np.sqrt(discriminant)) / 2
        r2 = (a - np.sqrt(discriminant)) / 2
        amplification = max(abs(r1), abs(r2))
    
    elif method.lower() in ["adams_bashforth_4", "ab4"]:
        # more complex characteristic equation
        # approximate for small z
        amplification = 1 + z + 3*z*z/8 + z*z*z/12
    
    elif method.lower() == "backward_euler":
        # implicit: yₙ₊₁ = yₙ + h*λ*yₙ₊₁
        # yₙ₊₁ = yₙ/(1 - h*λ)
        amplification = 1 / (1 - z)
    
    elif method.lower() == "trapezoidal":
        # implicit: yₙ₊₁ = yₙ + h*(fₙ + fₙ₊₁)/2
        amplification = (1 + z/2) / (1 - z/2)
    
    else:
        raise StabilityError(f"stability analysis not implemented for method: {method}")
    
    # compute spectral radius
    spectral_radius = abs(amplification)
    
    # for complex eigenvalue, check real part
    if isinstance(amplification, complex):
        max_eig = abs(amplification)
    else:
        max_eig = abs(amplification)
    
    # stability condition: |amplification| ≤ 1 for Re(λ) ≤ 0
    if lambda_val.real <= 0:
        stable = spectral_radius <= 1.0 + 1e-10  # small tolerance
        if stable:
            message = f"stable for λ={lambda_val}, h={step_size}"
        else:
            message = f"unstable for λ={lambda_val}, h={step_size} (spectral radius = {spectral_radius:.4f})"
    else:
        # for Re(λ) > 0, solution grows exponentially (unstable problem)
        stable = True  # method stability not applicable
        message = f"problem is inherently unstable (Re(λ) > 0)"
    
    return StabilityResult(
        stable=stable,
        spectral_radius=spectral_radius,
        max_eigenvalue=max_eig,
        method=method,
        message=message
    )


def truncation_error_estimate(method: str, h: float, derivative_bound: float) -> float:
    """
    estimate local truncation error for ode method
    
    uses theoretical error formulas:
    - euler: O(h²) per step → error ≈ h²·M₂/2
    - heun: O(h³) per step → error ≈ h³·M₃/6
    - rk4: O(h⁵) per step → error ≈ h⁵·M₅/180
    
    args:
        method: name of ode method
        h: step size
        derivative_bound: upper bound on relevant derivative
    
    returns:
        estimated local truncation error
    
    raises:
        ValueError: if method is not recognized
    """
    if method.lower() == "euler":
        return h*h * derivative_bound / 2
    elif method.lower() == "heun":
        return h*h*h * derivative_bound / 6
    elif method.lower() == "rk4":
        return h*h*h*h*h * derivative_bound / 180
    elif method.lower() in ["adams_bashforth_2", "ab2"]:
        return 5 * h*h*h * derivative_bound / 12
    elif method.lower() in ["adams_bashforth_4", "ab4"]:
        return 251 * h*h*h*h*h * derivative_bound / 720
    else:
        raise ValueError(f"truncation error formula not available for: {method}")


def global_error_estimate(local_error: float, x0: float, x_end: float, h: float) -> float:
    """
    estimate global error from local truncation error
    
    global error ≈ (x_end - x0) * local_error / h
    
    args:
        local_error: local truncation error per step
        x0: initial x value
        x_end: final x value
        h: step size
    
    returns:
        estimated global error
    """
    n_steps = (x_end - x0) / h
    return n_steps * local_error


def condition_number(f: Callable[[float], float], x: float, h: float = 1e-8) -> float:
    """
    estimate condition number of function evaluation
    
    measures sensitivity of f(x) to perturbations in x:
    κ = |x·f'(x) / f(x)|
    
    args:
        f: function to analyze
        x: point of evaluation
        h: step size for derivative
    
    returns:
        condition number (large values indicate ill-conditioning)
    """
    try:
        fx = f(x)
        if abs(fx) < 1e-15:
            return np.inf  # undefined
        
        # central difference for derivative
        f_plus = f(x + h)
        f_minus = f(x - h)
        fprime = (f_plus - f_minus) / (2 * h)
        
        kappa = abs(x * fprime / fx)
        return kappa
    except:
        return np.inf


def richardson_error_estimate(coarse: float, fine: float, order: int) -> float:
    """
    estimate error using richardson extrapolation
    
    if method has order p, then:
    error ≈ |fine - coarse| / (2^p - 1)
    
    args:
        coarse: result with step size h
        fine: result with step size h/2
        order: order of accuracy of method
    
    returns:
        estimated error in fine result
    """
    return abs(fine - coarse) / (2**order - 1)


def adaptive_step_controller(error: float, tol: float, h: float, 
                            order: int, safety: float = 0.9) -> float:
    """
    compute optimal step size for adaptive integration
    
    uses PI controller formula:
    h_new = h * safety * (tol / error)^(1/(order+1))
    
    args:
        error: current error estimate
        tol: desired tolerance
        h: current step size
        order: method order
        safety: safety factor (< 1)
    
    returns:
        new step size
    """
    if error <= 0 or not np.isfinite(error):
        return h  # keep current step
    
    exponent = 1.0 / (order + 1)
    factor = safety * (tol / error) ** exponent
    
    # limit step size changes
    factor = min(max(factor, 0.1), 5.0)
    
    return h * factor
