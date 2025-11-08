"""
high-level api for numerical integration, differentiation, and ode solving

provides simple interface: integrate(), differentiate(), solve_ode()
"""

from typing import Callable, Optional, Dict, Any, Union
import numpy as np

from numeric_integrator.integrators import (
    trapezoidal, simpson, midpoint, boole, romberg,
    adaptive_trapezoidal, adaptive_simpson, IntegrationResult
)
from numeric_integrator.differentiators import (
    forward_difference, backward_difference, central_difference,
    richardson_extrapolation, DerivativeResult
)
from numeric_integrator.ode_solvers import (
    euler, heun, rk4, rkf45, adams_bashforth, adams_moulton, ODESolution
)
from numeric_integrator.errors import IntegrationError, DifferentiationError, ODEError


def integrate(f: Callable[[float], float], a: float, b: float,
             method: str = "simpson", n: Optional[int] = None,
             **kwargs) -> IntegrationResult:
    """
    numerically integrate function f from a to b
    
    unified interface for all integration methods.
    
    available methods:
    - "trapezoidal": trapezoidal rule, O(h²)
    - "simpson": simpson's rule, O(h⁴)
    - "midpoint": midpoint rule, O(h²)
    - "boole": boole's rule, O(h⁶)
    - "romberg": romberg integration with extrapolation
    - "adaptive_trap": adaptive trapezoidal rule
    - "adaptive_simpson": adaptive simpson's rule (default for high accuracy)
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        method: integration method name
        n: number of subintervals (for non-adaptive methods)
        **kwargs: additional method-specific parameters
            - tol: tolerance (for adaptive/romberg)
            - max_iter: maximum iterations (for romberg)
            - max_depth: maximum recursion depth (for adaptive)
    
    returns:
        IntegrationResult with value, error estimate, and metadata
    
    raises:
        IntegrationError: if method is invalid or integration fails
    
    examples:
        >>> result = integrate(lambda x: x**2, 0, 1, method="simpson", n=100)
        >>> print(f"∫x² dx from 0 to 1 = {result.value:.6f} ± {result.error:.2e}")
        
        >>> result = integrate(lambda x: np.sin(x), 0, np.pi, method="adaptive_simpson", tol=1e-10)
        >>> print(f"integral = {result.value:.10f}")
    """
    method = method.lower()
    
    # determine default n if not specified
    if n is None:
        n = 100 if method in ["trapezoidal", "midpoint"] else 100
        if method == "simpson":
            n = 100 if n % 2 == 0 else 101
        elif method == "boole":
            n = 100 if n % 4 == 0 else 100
    
    try:
        if method == "trapezoidal":
            return trapezoidal(f, a, b, n)
        
        elif method == "simpson":
            if n % 2 != 0:
                n += 1  # ensure even
            return simpson(f, a, b, n)
        
        elif method == "midpoint":
            return midpoint(f, a, b, n)
        
        elif method == "boole":
            while n % 4 != 0:
                n += 1  # ensure divisible by 4
            return boole(f, a, b, n)
        
        elif method == "romberg":
            max_iter = kwargs.get("max_iter", 10)
            tol = kwargs.get("tol", 1e-10)
            return romberg(f, a, b, max_iter=max_iter, tol=tol)
        
        elif method in ["adaptive_trap", "adaptive_trapezoidal"]:
            tol = kwargs.get("tol", 1e-8)
            max_depth = kwargs.get("max_depth", 20)
            return adaptive_trapezoidal(f, a, b, tol=tol, max_depth=max_depth)
        
        elif method in ["adaptive_simpson", "adaptive"]:
            tol = kwargs.get("tol", 1e-10)
            max_depth = kwargs.get("max_depth", 20)
            return adaptive_simpson(f, a, b, tol=tol, max_depth=max_depth)
        
        else:
            raise IntegrationError(f"unknown integration method: {method}")
    
    except IntegrationError:
        raise
    except Exception as e:
        raise IntegrationError(f"integration failed with method '{method}': {e}")


def differentiate(f: Callable[[float], float], x0: float,
                 method: str = "central", h: Optional[float] = None,
                 **kwargs) -> DerivativeResult:
    """
    numerically differentiate function f at point x0
    
    unified interface for all differentiation methods.
    
    available methods:
    - "forward": forward difference, O(h)
    - "backward": backward difference, O(h)
    - "central": central difference, O(h²) (recommended)
    - "richardson": richardson extrapolation (high accuracy)
    
    args:
        f: function to differentiate
        x0: point at which to compute derivative
        method: differentiation method name
        h: step size (auto-selected if None)
        **kwargs: additional method-specific parameters
            - n_iter: number of iterations (for richardson)
    
    returns:
        DerivativeResult with value, error estimate, and metadata
    
    raises:
        DifferentiationError: if method is invalid or differentiation fails
    
    examples:
        >>> result = differentiate(lambda x: x**2, x0=3.0, method="central")
        >>> print(f"d/dx(x²) at x=3 = {result.value:.6f} (exact: 6.0)")
        
        >>> result = differentiate(np.sin, x0=0.0, method="richardson")
        >>> print(f"d/dx(sin(x)) at x=0 = {result.value:.10f} (exact: 1.0)")
    """
    method = method.lower()
    
    # auto-select step size if not specified
    if h is None:
        if method == "richardson":
            h = 1e-3
        elif method == "central":
            h = 1e-5
        else:
            h = 1e-5
    
    try:
        if method == "forward":
            return forward_difference(f, x0, h)
        
        elif method == "backward":
            return backward_difference(f, x0, h)
        
        elif method == "central":
            return central_difference(f, x0, h)
        
        elif method == "richardson":
            n_iter = kwargs.get("n_iter", 4)
            return richardson_extrapolation(f, x0, h, n_iter=n_iter)
        
        else:
            raise DifferentiationError(f"unknown differentiation method: {method}")
    
    except DifferentiationError:
        raise
    except Exception as e:
        raise DifferentiationError(f"differentiation failed with method '{method}': {e}")


def solve_ode(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
             method: str = "rk4", step: Optional[float] = None,
             **kwargs) -> ODESolution:
    """
    solve ordinary differential equation dy/dx = f(x, y)
    
    unified interface for all ode solvers.
    
    available methods:
    - "euler": euler's method, O(h) per step
    - "heun": heun's method (rk2), O(h²) per step
    - "rk4": classical runge-kutta, O(h⁴) per step (recommended)
    - "rkf45": adaptive runge-kutta-fehlberg (high accuracy, automatic step control)
    - "ab2", "ab3", "ab4": adams-bashforth predictor
    - "am2", "am3", "am4": adams-moulton predictor-corrector
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        method: ode solver method name
        step: step size (auto-selected if None)
        **kwargs: additional method-specific parameters
            - tol: tolerance (for rkf45)
            - h_min, h_max: step size bounds (for rkf45)
            - order: adams method order
    
    returns:
        ODESolution with x array, y array, and metadata
    
    raises:
        ODEError: if method is invalid or solving fails
    
    examples:
        >>> # solve dy/dx = -y with y(0) = 1
        >>> sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=2.0, method="rk4", step=0.01)
        >>> print(f"y(2) = {sol.y[-1]:.6f} (exact: {np.exp(-2):.6f})")
        
        >>> # van der pol oscillator with adaptive stepping
        >>> mu = 1.0
        >>> def van_der_pol(x, y):
        ...     return np.array([y[1], mu*(1 - y[0]**2)*y[1] - y[0]])
        >>> sol = solve_ode(van_der_pol, y0=[2.0, 0.0], x0=0.0, x_end=20.0, method="rkf45")
    """
    method = method.lower()
    
    # auto-select step size if not specified
    if step is None:
        if method == "euler":
            step = (x_end - x0) / 1000
        elif method in ["heun", "rk4"]:
            step = (x_end - x0) / 200
        elif method.startswith("ab") or method.startswith("am"):
            step = (x_end - x0) / 200
        else:
            step = 0.01
    
    try:
        if method == "euler":
            return euler(f, y0, x0, x_end, step)
        
        elif method == "heun":
            return heun(f, y0, x0, x_end, step)
        
        elif method == "rk4":
            return rk4(f, y0, x0, x_end, step)
        
        elif method == "rkf45":
            tol = kwargs.get("tol", 1e-6)
            h_init = kwargs.get("h_init", step)
            h_min = kwargs.get("h_min", 1e-8)
            h_max = kwargs.get("h_max", 1.0)
            return rkf45(f, y0, x0, x_end, tol=tol, h_init=h_init, h_min=h_min, h_max=h_max)
        
        elif method in ["ab2", "ab3", "ab4"]:
            order = int(method[-1])
            return adams_bashforth(f, y0, x0, x_end, step, order=order)
        
        elif method in ["am2", "am3", "am4"]:
            order = int(method[-1])
            return adams_moulton(f, y0, x0, x_end, step, order=order)
        
        elif method in ["adams_bashforth", "adams_moulton"]:
            order = kwargs.get("order", 4)
            if "bashforth" in method:
                return adams_bashforth(f, y0, x0, x_end, step, order=order)
            else:
                return adams_moulton(f, y0, x0, x_end, step, order=order)
        
        else:
            raise ODEError(f"unknown ode solver method: {method}")
    
    except ODEError:
        raise
    except Exception as e:
        raise ODEError(f"ode solving failed with method '{method}': {e}")


# convenience functions for common operations

def integrate_dataset(x: np.ndarray, y: np.ndarray, method: str = "trapezoidal") -> float:
    """
    integrate discrete dataset using specified method
    
    args:
        x: x coordinates (must be sorted)
        y: y coordinates
        method: integration method
    
    returns:
        approximate integral value
    """
    if method == "trapezoidal":
        return float(np.trapz(y, x))
    elif method == "simpson":
        if len(x) % 2 == 0:
            # simpson requires odd number of points
            return float(np.trapz(y, x))
        from scipy import integrate as scipy_integrate
        return float(scipy_integrate.simpson(y, x=x))
    else:
        return float(np.trapz(y, x))


def definite_integral(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10) -> float:
    """
    compute definite integral with high accuracy
    
    automatically selects best method and parameters.
    
    args:
        f: function to integrate
        a: lower bound
        b: upper bound
        tol: desired tolerance
    
    returns:
        integral value
    """
    result = integrate(f, a, b, method="adaptive_simpson", tol=tol)
    return result.value


def derivative_at(f: Callable[[float], float], x0: float, 
                 order: int = 1, h: Optional[float] = None) -> float:
    """
    compute derivative of specified order at point
    
    args:
        f: function to differentiate
        x0: point of evaluation
        order: derivative order (1 or 2)
        h: step size
    
    returns:
        derivative value
    """
    if order == 1:
        result = differentiate(f, x0, method="central", h=h)
        return result.value
    elif order == 2:
        from numeric_integrator.differentiators import second_derivative
        result = second_derivative(f, x0, h=h if h else 1e-4)
        return result.value
    else:
        raise ValueError(f"derivative order {order} not supported")
