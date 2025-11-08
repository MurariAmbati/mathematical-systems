"""
numeric_integrator: modular numerical integration and differential equation solver library

implements classical and modern methods for continuous function approximation and analysis.
"""

from numeric_integrator.api import integrate, differentiate, solve_ode
from numeric_integrator.integrators import (
    trapezoidal,
    simpson,
    midpoint,
    boole,
    romberg,
    adaptive_trapezoidal,
    adaptive_simpson,
)
from numeric_integrator.differentiators import (
    forward_difference,
    backward_difference,
    central_difference,
    richardson_extrapolation,
)
from numeric_integrator.ode_solvers import (
    euler,
    heun,
    rk4,
    rkf45,
    adams_bashforth,
    adams_moulton,
)
from numeric_integrator.analysis import (
    estimate_error,
    convergence_test,
    stability_analysis,
)

__version__ = "0.1.0"
__all__ = [
    "integrate",
    "differentiate",
    "solve_ode",
    "trapezoidal",
    "simpson",
    "midpoint",
    "boole",
    "romberg",
    "adaptive_trapezoidal",
    "adaptive_simpson",
    "forward_difference",
    "backward_difference",
    "central_difference",
    "richardson_extrapolation",
    "euler",
    "heun",
    "rk4",
    "rkf45",
    "adams_bashforth",
    "adams_moulton",
    "estimate_error",
    "convergence_test",
    "stability_analysis",
]
