"""
tests for api module

verifies high-level interface functions.
"""

import pytest
import numpy as np
from numeric_integrator import api, integrate, differentiate, solve_ode
from numeric_integrator.errors import IntegrationError, DifferentiationError, ODEError


class TestIntegrateAPI:
    """test high-level integrate() function"""
    
    def test_default_method(self):
        """test with default method (simpson)"""
        result = integrate(lambda x: x**2, 0, 1)
        assert abs(result.value - 1/3) < 1e-8
    
    def test_trapezoidal_method(self):
        """test with trapezoidal method"""
        result = integrate(lambda x: x**2, 0, 1, method="trapezoidal", n=100)
        assert result.method == "trapezoidal"
        assert abs(result.value - 1/3) < 1e-3
    
    def test_simpson_method(self):
        """test with simpson method"""
        result = integrate(np.sin, 0, np.pi, method="simpson", n=100)
        assert result.method == "simpson"
        assert abs(result.value - 2.0) < 1e-10
    
    def test_midpoint_method(self):
        """test with midpoint method"""
        result = integrate(lambda x: x**3, 0, 2, method="midpoint", n=100)
        assert result.method == "midpoint"
        assert abs(result.value - 4.0) < 1e-3
    
    def test_boole_method(self):
        """test with boole method"""
        result = integrate(lambda x: x**4, 0, 1, method="boole", n=100)
        assert result.method == "boole"
        assert abs(result.value - 0.2) < 1e-10
    
    def test_romberg_method(self):
        """test with romberg method"""
        result = integrate(lambda x: np.exp(x), 0, 1, method="romberg", tol=1e-10)
        assert result.method == "romberg"
        expected = np.e - 1
        assert abs(result.value - expected) < 1e-10
    
    def test_adaptive_trapezoidal(self):
        """test with adaptive trapezoidal"""
        result = integrate(lambda x: 1/(1+x**2), 0, 1, 
                         method="adaptive_trap", tol=1e-8)
        assert result.method == "adaptive_trapezoidal"
        assert abs(result.value - np.pi/4) < 1e-7
    
    def test_adaptive_simpson(self):
        """test with adaptive simpson"""
        result = integrate(lambda x: np.sin(x), 0, np.pi, 
                         method="adaptive_simpson", tol=1e-10)
        assert result.method == "adaptive_simpson"
        assert abs(result.value - 2.0) < 1e-10
    
    def test_auto_even_n_simpson(self):
        """simpson should auto-adjust n to be even"""
        result = integrate(lambda x: x, 0, 1, method="simpson", n=99)
        assert result.method == "simpson"
    
    def test_auto_divisible_n_boole(self):
        """boole should auto-adjust n to be divisible by 4"""
        result = integrate(lambda x: x, 0, 1, method="boole", n=99)
        assert result.method == "boole"
    
    def test_unknown_method(self):
        """test with unknown method"""
        with pytest.raises(IntegrationError, match="unknown"):
            integrate(lambda x: x, 0, 1, method="unknown_method")


class TestDifferentiateAPI:
    """test high-level differentiate() function"""
    
    def test_default_method(self):
        """test with default method (central)"""
        result = differentiate(lambda x: x**2, x0=3.0)
        assert abs(result.value - 6.0) < 1e-6
        assert result.method == "central_difference"
    
    def test_forward_method(self):
        """test with forward difference"""
        result = differentiate(np.sin, x0=0.0, method="forward", h=1e-5)
        assert result.method == "forward_difference"
        assert abs(result.value - 1.0) < 1e-4
    
    def test_backward_method(self):
        """test with backward difference"""
        result = differentiate(np.cos, x0=np.pi/3, method="backward", h=1e-5)
        expected = -np.sin(np.pi/3)
        assert result.method == "backward_difference"
        assert abs(result.value - expected) < 1e-4
    
    def test_central_method(self):
        """test with central difference"""
        result = differentiate(lambda x: x**3, x0=2.0, method="central", h=1e-5)
        expected = 3 * 2**2
        assert abs(result.value - expected) < 1e-6
    
    def test_richardson_method(self):
        """test with richardson extrapolation"""
        result = differentiate(np.exp, x0=1.0, method="richardson", h=1e-3, n_iter=4)
        assert result.method == "richardson_extrapolation"
        assert abs(result.value - np.e) < 1e-10
    
    def test_auto_step_size(self):
        """test automatic step size selection"""
        result = differentiate(np.sin, x0=0.0)
        assert abs(result.value - 1.0) < 1e-6
    
    def test_unknown_method(self):
        """test with unknown method"""
        with pytest.raises(DifferentiationError, match="unknown"):
            differentiate(lambda x: x, x0=0, method="unknown_method")


class TestSolveODEAPI:
    """test high-level solve_ode() function"""
    
    def test_default_method(self):
        """test with default method (rk4)"""
        sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0, step=0.01)
        assert sol.method == "rk4"
        assert sol.success
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 1e-5
    
    def test_euler_method(self):
        """test with euler method"""
        sol = solve_ode(lambda x, y: y, y0=1.0, x0=0.0, x_end=1.0, 
                       method="euler", step=0.01)
        assert sol.method == "euler"
        assert sol.success
    
    def test_heun_method(self):
        """test with heun method"""
        sol = solve_ode(lambda x, y: -2*y, y0=1.0, x0=0.0, x_end=1.0,
                       method="heun", step=0.01)
        assert sol.method == "heun"
        assert sol.success
    
    def test_rk4_method(self):
        """test with rk4 method"""
        sol = solve_ode(lambda x, y: x + y, y0=1.0, x0=0.0, x_end=1.0,
                       method="rk4", step=0.01)
        assert sol.method == "rk4"
        assert sol.success
    
    def test_rkf45_method(self):
        """test with adaptive rkf45 method"""
        sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0,
                       method="rkf45", tol=1e-6)
        assert sol.method == "rkf45"
        assert sol.success
    
    def test_adams_bashforth(self):
        """test with adams-bashforth methods"""
        for order in [2, 3, 4]:
            sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0,
                           method=f"ab{order}", step=0.01)
            assert f"adams_bashforth_{order}" in sol.method
            assert sol.success
    
    def test_adams_moulton(self):
        """test with adams-moulton methods"""
        for order in [2, 3, 4]:
            sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0,
                           method=f"am{order}", step=0.01)
            assert f"adams_moulton_{order}" in sol.method
            assert sol.success
    
    def test_auto_step_size(self):
        """test automatic step size selection"""
        sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0)
        assert sol.success
    
    def test_unknown_method(self):
        """test with unknown method"""
        with pytest.raises(ODEError, match="unknown"):
            solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0, 
                     method="unknown_method")


class TestDefiniteIntegral:
    """test convenience function definite_integral()"""
    
    def test_polynomial(self):
        """integrate polynomial"""
        result = api.definite_integral(lambda x: x**3, 0, 2, tol=1e-10)
        assert abs(result - 4.0) < 1e-10
    
    def test_trig(self):
        """integrate trig function"""
        result = api.definite_integral(np.cos, 0, np.pi/2, tol=1e-12)
        assert abs(result - 1.0) < 1e-12
    
    def test_returns_float(self):
        """should return float, not result object"""
        result = api.definite_integral(lambda x: x, 0, 1)
        assert isinstance(result, float)


class TestDerivativeAt:
    """test convenience function derivative_at()"""
    
    def test_first_derivative(self):
        """compute first derivative"""
        result = api.derivative_at(lambda x: x**2, x0=5.0, order=1)
        assert abs(result - 10.0) < 1e-6
    
    def test_second_derivative(self):
        """compute second derivative"""
        result = api.derivative_at(lambda x: x**3, x0=2.0, order=2)
        expected = 12.0  # d²/dx²(x³) = 6x at x=2
        assert abs(result - expected) < 1e-3
    
    def test_returns_float(self):
        """should return float, not result object"""
        result = api.derivative_at(lambda x: x**2, x0=1.0)
        assert isinstance(result, float)
    
    def test_invalid_order(self):
        """test with unsupported order"""
        with pytest.raises(ValueError, match="not supported"):
            api.derivative_at(lambda x: x, x0=0, order=3)


class TestAPIEdgeCases:
    """test edge cases in api"""
    
    def test_integrate_with_kwargs(self):
        """test passing kwargs to integrate"""
        result = integrate(lambda x: x**2, 0, 1, method="romberg", 
                         max_iter=15, tol=1e-12)
        assert abs(result.value - 1/3) < 1e-12
    
    def test_differentiate_with_kwargs(self):
        """test passing kwargs to differentiate"""
        result = differentiate(np.sin, x0=0.0, method="richardson",
                             h=1e-3, n_iter=5)
        assert abs(result.value - 1.0) < 1e-12
    
    def test_solve_ode_with_kwargs(self):
        """test passing kwargs to solve_ode"""
        sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0,
                       method="rkf45", tol=1e-8, h_min=1e-6, h_max=0.5)
        assert sol.success
    
    def test_integrate_none_n(self):
        """test with n=None (auto-select)"""
        result = integrate(lambda x: x, 0, 1, method="simpson")
        assert result.method == "simpson"
    
    def test_differentiate_none_h(self):
        """test with h=None (auto-select)"""
        result = differentiate(lambda x: x, x0=1.0, method="central")
        assert abs(result.value - 1.0) < 1e-6
    
    def test_solve_ode_none_step(self):
        """test with step=None (auto-select)"""
        sol = solve_ode(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0, method="rk4")
        assert sol.success
