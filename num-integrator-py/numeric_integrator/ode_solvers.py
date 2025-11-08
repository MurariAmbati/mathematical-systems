"""
ordinary differential equation (ode) solvers

implements classical and modern methods for solving dy/dx = f(x, y).
"""

from typing import Callable, Tuple, List
import numpy as np
from dataclasses import dataclass

from numeric_integrator.errors import ODEError


@dataclass
class ODESolution:
    """solution of ordinary differential equation"""
    x: np.ndarray  # independent variable values
    y: np.ndarray  # dependent variable values
    n_evaluations: int
    method: str
    success: bool = True
    message: str = "integration successful"


def euler(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
          step: float = 0.01) -> ODESolution:
    """
    euler's method for solving dy/dx = f(x, y)
    
    first-order explicit method: yₙ₊₁ = yₙ + h·f(xₙ, yₙ)
    
    accuracy: O(h) per step, O(h) global
    stability: conditionally stable, requires small h
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        step: step size h
    
    returns:
        ODESolution with x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid or integration fails
    """
    if step <= 0:
        raise ODEError(f"step size must be positive, got {step}")
    if not np.isfinite(x0) or not np.isfinite(x_end) or not np.isfinite(y0):
        raise ODEError(f"initial conditions must be finite: x0={x0}, x_end={x_end}, y0={y0}")
    if x0 >= x_end:
        raise ODEError(f"x0 must be less than x_end: x0={x0}, x_end={x_end}")
    
    # setup
    n_steps = int(np.ceil((x_end - x0) / step)) + 1
    x_values = np.linspace(x0, x_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0
    
    n_evals = 0
    
    # euler integration
    for i in range(n_steps - 1):
        x_curr = x_values[i]
        y_curr = y_values[i]
        h = x_values[i+1] - x_values[i]
        
        try:
            k1 = f(x_curr, y_curr)
            n_evals += 1
        except Exception as e:
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="euler",
                success=False,
                message=f"function evaluation failed at step {i}: {e}"
            )
        
        if not np.isfinite(k1):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="euler",
                success=False,
                message=f"non-finite derivative at step {i}"
            )
        
        y_values[i+1] = y_curr + h * k1
        
        if not np.isfinite(y_values[i+1]):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="euler",
                success=False,
                message=f"solution became non-finite at step {i}"
            )
    
    return ODESolution(
        x=x_values,
        y=y_values,
        n_evaluations=n_evals,
        method="euler"
    )


def heun(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
         step: float = 0.01) -> ODESolution:
    """
    heun's method (improved euler) for solving dy/dx = f(x, y)
    
    second-order runge-kutta method (rk2):
    k₁ = f(xₙ, yₙ)
    k₂ = f(xₙ + h, yₙ + h·k₁)
    yₙ₊₁ = yₙ + h·(k₁ + k₂)/2
    
    accuracy: O(h²) per step, O(h²) global
    stability: better than euler, still conditionally stable
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        step: step size h
    
    returns:
        ODESolution with x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid
    """
    if step <= 0:
        raise ODEError(f"step size must be positive, got {step}")
    if not np.isfinite(x0) or not np.isfinite(x_end) or not np.isfinite(y0):
        raise ODEError(f"initial conditions must be finite")
    if x0 >= x_end:
        raise ODEError(f"x0 must be less than x_end")
    
    n_steps = int(np.ceil((x_end - x0) / step)) + 1
    x_values = np.linspace(x0, x_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0
    
    n_evals = 0
    
    for i in range(n_steps - 1):
        x_curr = x_values[i]
        y_curr = y_values[i]
        h = x_values[i+1] - x_values[i]
        
        try:
            k1 = f(x_curr, y_curr)
            k2 = f(x_curr + h, y_curr + h * k1)
            n_evals += 2
        except Exception as e:
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="heun",
                success=False,
                message=f"function evaluation failed at step {i}: {e}"
            )
        
        if not np.isfinite(k1) or not np.isfinite(k2):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="heun",
                success=False,
                message=f"non-finite derivative at step {i}"
            )
        
        y_values[i+1] = y_curr + h * (k1 + k2) / 2
        
        if not np.isfinite(y_values[i+1]):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="heun",
                success=False,
                message=f"solution became non-finite at step {i}"
            )
    
    return ODESolution(
        x=x_values,
        y=y_values,
        n_evaluations=n_evals,
        method="heun"
    )


def rk4(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
        step: float = 0.01) -> ODESolution:
    """
    classical fourth-order runge-kutta method (rk4)
    
    the standard workhorse ode solver:
    k₁ = f(xₙ, yₙ)
    k₂ = f(xₙ + h/2, yₙ + h·k₁/2)
    k₃ = f(xₙ + h/2, yₙ + h·k₂/2)
    k₄ = f(xₙ + h, yₙ + h·k₃)
    yₙ₊₁ = yₙ + h·(k₁ + 2k₂ + 2k₃ + k₄)/6
    
    accuracy: O(h⁴) per step, O(h⁴) global
    stability: good stability, widely used
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        step: step size h
    
    returns:
        ODESolution with x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid
    """
    if step <= 0:
        raise ODEError(f"step size must be positive, got {step}")
    if not np.isfinite(x0) or not np.isfinite(x_end) or not np.isfinite(y0):
        raise ODEError(f"initial conditions must be finite")
    if x0 >= x_end:
        raise ODEError(f"x0 must be less than x_end")
    
    n_steps = int(np.ceil((x_end - x0) / step)) + 1
    x_values = np.linspace(x0, x_end, n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0
    
    n_evals = 0
    
    for i in range(n_steps - 1):
        x_curr = x_values[i]
        y_curr = y_values[i]
        h = x_values[i+1] - x_values[i]
        
        try:
            k1 = f(x_curr, y_curr)
            k2 = f(x_curr + h/2, y_curr + h * k1 / 2)
            k3 = f(x_curr + h/2, y_curr + h * k2 / 2)
            k4 = f(x_curr + h, y_curr + h * k3)
            n_evals += 4
        except Exception as e:
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="rk4",
                success=False,
                message=f"function evaluation failed at step {i}: {e}"
            )
        
        if not all(np.isfinite([k1, k2, k3, k4])):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="rk4",
                success=False,
                message=f"non-finite derivative at step {i}"
            )
        
        y_values[i+1] = y_curr + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        
        if not np.isfinite(y_values[i+1]):
            return ODESolution(
                x=x_values[:i+1],
                y=y_values[:i+1],
                n_evaluations=n_evals,
                method="rk4",
                success=False,
                message=f"solution became non-finite at step {i}"
            )
    
    return ODESolution(
        x=x_values,
        y=y_values,
        n_evaluations=n_evals,
        method="rk4"
    )


def rkf45(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
          tol: float = 1e-6, h_init: float = 0.01, h_min: float = 1e-8,
          h_max: float = 1.0) -> ODESolution:
    """
    runge-kutta-fehlberg adaptive method (rkf45)
    
    uses embedded 4th/5th order runge-kutta pair to estimate local error
    and automatically adjust step size for efficiency and accuracy.
    
    accuracy: O(h⁴) with automatic error control
    stability: adaptive step size ensures stability
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        tol: error tolerance for step size control
        h_init: initial step size
        h_min: minimum allowed step size
        h_max: maximum allowed step size
    
    returns:
        ODESolution with adaptive x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid or step size becomes too small
    """
    if tol <= 0:
        raise ODEError(f"tolerance must be positive, got {tol}")
    if h_init <= 0:
        raise ODEError(f"initial step size must be positive, got {h_init}")
    if h_min <= 0 or h_max <= h_min:
        raise ODEError(f"invalid step size bounds: h_min={h_min}, h_max={h_max}")
    if not np.isfinite(x0) or not np.isfinite(x_end) or not np.isfinite(y0):
        raise ODEError(f"initial conditions must be finite")
    if x0 >= x_end:
        raise ODEError(f"x0 must be less than x_end")
    
    # butcher tableau coefficients for rkf45
    a = [0, 1/4, 3/8, 12/13, 1, 1/2]
    b = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    c4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]  # 4th order
    c5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]  # 5th order
    
    x_list = [x0]
    y_list = [y0]
    h = h_init
    n_evals = 0
    
    x_curr = x0
    y_curr = y0
    
    while x_curr < x_end:
        # don't overshoot
        if x_curr + h > x_end:
            h = x_end - x_curr
        
        # compute k values
        try:
            k = np.zeros(6)
            k[0] = f(x_curr, y_curr)
            k[1] = f(x_curr + a[1]*h, y_curr + h*b[1][0]*k[0])
            k[2] = f(x_curr + a[2]*h, y_curr + h*(b[2][0]*k[0] + b[2][1]*k[1]))
            k[3] = f(x_curr + a[3]*h, y_curr + h*(b[3][0]*k[0] + b[3][1]*k[1] + b[3][2]*k[2]))
            k[4] = f(x_curr + a[4]*h, y_curr + h*(b[4][0]*k[0] + b[4][1]*k[1] + b[4][2]*k[2] + b[4][3]*k[3]))
            k[5] = f(x_curr + a[5]*h, y_curr + h*(b[5][0]*k[0] + b[5][1]*k[1] + b[5][2]*k[2] + b[5][3]*k[3] + b[5][4]*k[4]))
            n_evals += 6
        except Exception as e:
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method="rkf45",
                success=False,
                message=f"function evaluation failed: {e}"
            )
        
        if not np.all(np.isfinite(k)):
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method="rkf45",
                success=False,
                message="non-finite derivative encountered"
            )
        
        # 4th and 5th order estimates
        y4 = y_curr + h * sum(c4[i] * k[i] for i in range(6))
        y5 = y_curr + h * sum(c5[i] * k[i] for i in range(6))
        
        # error estimate
        error = abs(y5 - y4)
        
        # accept or reject step
        if error <= tol or h <= h_min:
            # accept
            x_curr += h
            y_curr = y5  # use higher-order estimate
            x_list.append(x_curr)
            y_list.append(y_curr)
            
            if not np.isfinite(y_curr):
                return ODESolution(
                    x=np.array(x_list),
                    y=np.array(y_list),
                    n_evaluations=n_evals,
                    method="rkf45",
                    success=False,
                    message="solution became non-finite"
                )
        
        # adjust step size
        if error > 0:
            s = 0.84 * (tol / error) ** 0.25  # safety factor
            h = h * min(max(s, 0.1), 4.0)  # limit change
            h = min(max(h, h_min), h_max)
        
        # safety check
        if h < h_min:
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method="rkf45",
                success=False,
                message=f"step size fell below minimum: h={h}, h_min={h_min}"
            )
    
    return ODESolution(
        x=np.array(x_list),
        y=np.array(y_list),
        n_evaluations=n_evals,
        method="rkf45"
    )


def adams_bashforth(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
                    step: float = 0.01, order: int = 4) -> ODESolution:
    """
    adams-bashforth multi-step predictor method
    
    uses previous function evaluations to extrapolate next value.
    requires initial values from single-step method (uses rk4).
    
    order 2: yₙ₊₁ = yₙ + h·(3f(xₙ,yₙ) - f(xₙ₋₁,yₙ₋₁))/2
    order 4: yₙ₊₁ = yₙ + h·(55fₙ - 59fₙ₋₁ + 37fₙ₋₂ - 9fₙ₋₃)/24
    
    accuracy: O(h^(order+1))
    stability: conditionally stable, requires good initial values
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        step: step size h
        order: method order (2, 3, or 4)
    
    returns:
        ODESolution with x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid
    """
    if order not in [2, 3, 4]:
        raise ODEError(f"order must be 2, 3, or 4, got {order}")
    if step <= 0:
        raise ODEError(f"step size must be positive, got {step}")
    
    # use rk4 to generate initial values
    init_sol = rk4(f, y0, x0, x0 + (order-1)*step, step)
    if not init_sol.success:
        return init_sol
    
    x_list = list(init_sol.x)
    y_list = list(init_sol.y)
    
    # store previous derivatives
    f_hist = [f(x_list[i], y_list[i]) for i in range(order)]
    n_evals = init_sol.n_evaluations + order
    
    # adams-bashforth coefficients
    ab_coeff = {
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24]
    }
    coeff = ab_coeff[order]
    
    # continue integration
    n_steps = int(np.ceil((x_end - x0) / step)) + 1
    x_curr = x_list[-1]
    
    while x_curr < x_end:
        h = min(step, x_end - x_curr)
        
        # adams-bashforth formula
        y_new = y_list[-1] + h * sum(coeff[i] * f_hist[-(i+1)] for i in range(order))
        x_new = x_curr + h
        
        if not np.isfinite(y_new):
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method=f"adams_bashforth_{order}",
                success=False,
                message="solution became non-finite"
            )
        
        x_list.append(x_new)
        y_list.append(y_new)
        
        # update derivative history
        try:
            f_new = f(x_new, y_new)
            n_evals += 1
        except Exception as e:
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method=f"adams_bashforth_{order}",
                success=False,
                message=f"function evaluation failed: {e}"
            )
        
        f_hist.append(f_new)
        if len(f_hist) > order:
            f_hist.pop(0)
        
        x_curr = x_new
    
    return ODESolution(
        x=np.array(x_list),
        y=np.array(y_list),
        n_evaluations=n_evals,
        method=f"adams_bashforth_{order}"
    )


def adams_moulton(f: Callable[[float, float], float], y0: float, x0: float, x_end: float,
                  step: float = 0.01, order: int = 4) -> ODESolution:
    """
    adams-moulton predictor-corrector method
    
    combines adams-bashforth predictor with adams-moulton corrector (implicit).
    more stable than pure adams-bashforth.
    
    predictor: yₚ = adams-bashforth
    corrector: yₙ₊₁ = yₙ + h·(9fₙ₊₁ + 19fₙ - 5fₙ₋₁ + fₙ₋₂)/24 (order 4)
    
    accuracy: O(h^(order+1))
    stability: better than adams-bashforth
    
    args:
        f: derivative function dy/dx = f(x, y)
        y0: initial condition y(x0)
        x0: initial x value
        x_end: final x value
        step: step size h
        order: method order (2, 3, or 4)
    
    returns:
        ODESolution with x values, y values, and metadata
    
    raises:
        ODEError: if parameters are invalid
    """
    if order not in [2, 3, 4]:
        raise ODEError(f"order must be 2, 3, or 4, got {order}")
    if step <= 0:
        raise ODEError(f"step size must be positive, got {step}")
    
    # use rk4 for initial values
    init_sol = rk4(f, y0, x0, x0 + (order-1)*step, step)
    if not init_sol.success:
        return init_sol
    
    x_list = list(init_sol.x)
    y_list = list(init_sol.y)
    
    # derivative history
    f_hist = [f(x_list[i], y_list[i]) for i in range(order)]
    n_evals = init_sol.n_evaluations + order
    
    # adams-bashforth coefficients (predictor)
    ab_coeff = {
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24]
    }
    
    # adams-moulton coefficients (corrector)
    am_coeff = {
        2: [1/2, 1/2],
        3: [5/12, 8/12, -1/12],
        4: [9/24, 19/24, -5/24, 1/24]
    }
    
    ab = ab_coeff[order]
    am = am_coeff[order]
    
    x_curr = x_list[-1]
    
    while x_curr < x_end:
        h = min(step, x_end - x_curr)
        
        # predictor (adams-bashforth)
        y_pred = y_list[-1] + h * sum(ab[i] * f_hist[-(i+1)] for i in range(order))
        x_new = x_curr + h
        
        # corrector (adams-moulton)
        try:
            f_pred = f(x_new, y_pred)
            n_evals += 1
        except Exception as e:
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method=f"adams_moulton_{order}",
                success=False,
                message=f"function evaluation failed: {e}"
            )
        
        # build corrector with predicted value
        f_corr = [f_pred] + f_hist[-(order-1):]
        y_corr = y_list[-1] + h * sum(am[i] * f_corr[i] for i in range(order))
        
        if not np.isfinite(y_corr):
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method=f"adams_moulton_{order}",
                success=False,
                message="solution became non-finite"
            )
        
        x_list.append(x_new)
        y_list.append(y_corr)
        
        # update derivative history with corrected value
        try:
            f_new = f(x_new, y_corr)
            n_evals += 1
        except Exception as e:
            return ODESolution(
                x=np.array(x_list),
                y=np.array(y_list),
                n_evaluations=n_evals,
                method=f"adams_moulton_{order}",
                success=False,
                message=f"function evaluation failed: {e}"
            )
        
        f_hist.append(f_new)
        if len(f_hist) > order:
            f_hist.pop(0)
        
        x_curr = x_new
    
    return ODESolution(
        x=np.array(x_list),
        y=np.array(y_list),
        n_evaluations=n_evals,
        method=f"adams_moulton_{order}"
    )
