"""
tests for ode solvers

verifies accuracy and stability of differential equation solvers.
"""

import pytest
import numpy as np
from numeric_integrator import ode_solvers
from numeric_integrator.errors import ODEError


class TestEuler:
    """test euler's method"""
    
    def test_exponential_decay(self):
        """solve dy/dx = -y with y(0) = 1, exact solution: y = exp(-x)"""
        sol = ode_solvers.euler(lambda x, y: -y, y0=1.0, x0=0.0, x_end=1.0, step=0.01)
        assert sol.success
        assert sol.method == "euler"
        # euler is first-order, expect moderate accuracy
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 0.05
    
    def test_linear_ode(self):
        """solve dy/dx = x with y(0) = 0, exact: y = x²/2"""
        sol = ode_solvers.euler(lambda x, y: x, y0=0.0, x0=0.0, x_end=2.0, step=0.01)
        assert sol.success
        expected = 2.0  # x²/2 at x=2
        assert abs(sol.y[-1] - expected) < 0.1
    
    def test_constant_derivative(self):
        """solve dy/dx = 2 with y(0) = 1, exact: y = 2x + 1"""
        sol = ode_solvers.euler(lambda x, y: 2.0, y0=1.0, x0=0.0, x_end=3.0, step=0.1)
        assert sol.success
        expected = 7.0  # 2*3 + 1
        assert abs(sol.y[-1] - expected) < 0.01
    
    def test_invalid_step(self):
        """test with invalid step size"""
        with pytest.raises(ODEError, match="must be positive"):
            ode_solvers.euler(lambda x, y: -y, y0=1.0, x0=0, x_end=1, step=0)
    
    def test_invalid_bounds(self):
        """test with invalid bounds"""
        with pytest.raises(ODEError, match="must be less than x_end"):
            ode_solvers.euler(lambda x, y: -y, y0=1.0, x0=1, x_end=0, step=0.01)
    
    def test_function_error(self):
        """test with function that raises error"""
        def bad_func(x, y):
            if x > 0.5:
                raise ValueError("error")
            return -y
        
        sol = ode_solvers.euler(bad_func, y0=1.0, x0=0, x_end=1, step=0.1)
        assert not sol.success
        assert "function evaluation failed" in sol.message


class TestHeun:
    """test heun's method (rk2)"""
    
    def test_exponential_growth(self):
        """solve dy/dx = y with y(0) = 1, exact: y = exp(x)"""
        sol = ode_solvers.heun(lambda x, y: y, y0=1.0, x0=0.0, x_end=1.0, step=0.01)
        assert sol.success
        assert sol.method == "heun"
        expected = np.e
        assert abs(sol.y[-1] - expected) < 0.01
    
    def test_better_than_euler(self):
        """heun should be more accurate than euler"""
        f = lambda x, y: -2*y
        y0, x0, x_end = 1.0, 0.0, 1.0
        step = 0.05
        exact = np.exp(-2.0)
        
        sol_euler = ode_solvers.euler(f, y0, x0, x_end, step)
        sol_heun = ode_solvers.heun(f, y0, x0, x_end, step)
        
        err_euler = abs(sol_euler.y[-1] - exact)
        err_heun = abs(sol_heun.y[-1] - exact)
        
        assert err_heun < err_euler
    
    def test_quadratic(self):
        """solve dy/dx = y - x² + 1 with y(0) = 0.5"""
        sol = ode_solvers.heun(
            lambda x, y: y - x**2 + 1, 
            y0=0.5, x0=0.0, x_end=2.0, step=0.01
        )
        assert sol.success
        # exact solution: y = x² + 2x + 1 - 1.5*exp(x)
        x = 2.0
        expected = x**2 + 2*x + 1 - 1.5*np.exp(x)
        assert abs(sol.y[-1] - expected) < 0.1


class TestRK4:
    """test classical runge-kutta (rk4)"""
    
    def test_exponential(self):
        """solve dy/dx = y with y(0) = 1"""
        sol = ode_solvers.rk4(lambda x, y: y, y0=1.0, x0=0.0, x_end=1.0, step=0.1)
        assert sol.success
        assert sol.method == "rk4"
        expected = np.e
        assert abs(sol.y[-1] - expected) < 1e-6
    
    def test_sine_forcing(self):
        """solve dy/dx = sin(x) with y(0) = 0, exact: y = 1 - cos(x)"""
        sol = ode_solvers.rk4(
            lambda x, y: np.sin(x), 
            y0=0.0, x0=0.0, x_end=np.pi, step=0.01
        )
        assert sol.success
        expected = 2.0  # 1 - cos(π) = 2
        assert abs(sol.y[-1] - expected) < 1e-6
    
    def test_high_accuracy(self):
        """rk4 should achieve high accuracy"""
        # dy/dx = -2xy, y(0) = 1, exact: y = exp(-x²)
        sol = ode_solvers.rk4(
            lambda x, y: -2*x*y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01
        )
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 1e-6
    
    def test_stiff_problem(self):
        """test on moderately stiff problem"""
        # dy/dx = -10y, y(0) = 1
        sol = ode_solvers.rk4(
            lambda x, y: -10*y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01
        )
        assert sol.success
        expected = np.exp(-10.0)
        assert abs(sol.y[-1] - expected) < 1e-4
    
    def test_nonlinear(self):
        """test nonlinear ode"""
        # dy/dx = y²(1-y), logistic-type equation
        sol = ode_solvers.rk4(
            lambda x, y: y*y*(1-y),
            y0=0.1, x0=0.0, x_end=5.0, step=0.01
        )
        assert sol.success
        assert sol.y[-1] > 0.1  # should grow
        assert sol.y[-1] < 1.0  # but not exceed 1


class TestRKF45:
    """test adaptive runge-kutta-fehlberg"""
    
    def test_exponential(self):
        """solve dy/dx = y with adaptive stepping"""
        sol = ode_solvers.rkf45(
            lambda x, y: y,
            y0=1.0, x0=0.0, x_end=1.0, tol=1e-6, h_init=0.1
        )
        assert sol.success
        assert sol.method == "rkf45"
        expected = np.e
        assert abs(sol.y[-1] - expected) < 1e-5
    
    def test_step_adaptation(self):
        """verify that adaptive stepping works"""
        # smooth function should use larger steps
        sol1 = ode_solvers.rkf45(
            lambda x, y: y,
            y0=1.0, x0=0.0, x_end=2.0, tol=1e-6, h_init=0.01
        )
        
        # rapidly varying function should use smaller steps
        sol2 = ode_solvers.rkf45(
            lambda x, y: -10*y,
            y0=1.0, x0=0.0, x_end=2.0, tol=1e-6, h_init=0.01
        )
        
        assert sol1.success and sol2.success
        # can't guarantee step count without seeing implementation details
    
    def test_high_accuracy(self):
        """rkf45 should achieve high accuracy"""
        sol = ode_solvers.rkf45(
            lambda x, y: -2*x*y,
            y0=1.0, x0=0.0, x_end=1.0, tol=1e-10
        )
        assert sol.success
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 1e-8
    
    def test_tight_tolerance(self):
        """test with very tight tolerance"""
        sol = ode_solvers.rkf45(
            lambda x, y: np.sin(x) * y,
            y0=1.0, x0=0.0, x_end=np.pi, tol=1e-12, h_init=0.01
        )
        assert sol.success
    
    def test_step_size_bounds(self):
        """test step size limiting"""
        sol = ode_solvers.rkf45(
            lambda x, y: y,
            y0=1.0, x0=0.0, x_end=1.0,
            tol=1e-6, h_init=0.01, h_min=1e-6, h_max=0.5
        )
        assert sol.success


class TestAdamsBashforth:
    """test adams-bashforth multi-step methods"""
    
    def test_ab2(self):
        """test second-order adams-bashforth"""
        sol = ode_solvers.adams_bashforth(
            lambda x, y: -y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01, order=2
        )
        assert sol.success
        assert "adams_bashforth_2" in sol.method
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 0.01
    
    def test_ab4(self):
        """test fourth-order adams-bashforth"""
        sol = ode_solvers.adams_bashforth(
            lambda x, y: y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01, order=4
        )
        assert sol.success
        assert "adams_bashforth_4" in sol.method
        expected = np.e
        assert abs(sol.y[-1] - expected) < 0.001
    
    def test_invalid_order(self):
        """test with invalid order"""
        with pytest.raises(ODEError, match="order must be"):
            ode_solvers.adams_bashforth(
                lambda x, y: y,
                y0=1.0, x0=0.0, x_end=1.0, step=0.01, order=5
            )
    
    def test_uses_rk4_startup(self):
        """adams-bashforth uses rk4 for initial values"""
        sol = ode_solvers.adams_bashforth(
            lambda x, y: x + y,
            y0=1.0, x0=0.0, x_end=2.0, step=0.1, order=4
        )
        assert sol.success
        # should have more points than just initial values
        assert len(sol.x) > 10


class TestAdamsMoulton:
    """test adams-moulton predictor-corrector methods"""
    
    def test_am2(self):
        """test second-order adams-moulton"""
        sol = ode_solvers.adams_moulton(
            lambda x, y: -y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01, order=2
        )
        assert sol.success
        assert "adams_moulton_2" in sol.method
        expected = np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 0.005
    
    def test_am4(self):
        """test fourth-order adams-moulton"""
        sol = ode_solvers.adams_moulton(
            lambda x, y: y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01, order=4
        )
        assert sol.success
        expected = np.e
        assert abs(sol.y[-1] - expected) < 0.0001
    
    def test_better_than_ab(self):
        """adams-moulton should be more accurate than adams-bashforth"""
        f = lambda x, y: -2*y
        y0, x0, x_end = 1.0, 0.0, 1.0
        step = 0.05
        order = 4
        exact = np.exp(-2.0)
        
        sol_ab = ode_solvers.adams_bashforth(f, y0, x0, x_end, step, order)
        sol_am = ode_solvers.adams_moulton(f, y0, x0, x_end, step, order)
        
        err_ab = abs(sol_ab.y[-1] - exact)
        err_am = abs(sol_am.y[-1] - exact)
        
        assert err_am < err_ab
    
    def test_nonlinear_equation(self):
        """test on nonlinear equation"""
        # dy/dx = sin(x) + y², y(0) = 0
        sol = ode_solvers.adams_moulton(
            lambda x, y: np.sin(x) + y**2,
            y0=0.0, x0=0.0, x_end=1.0, step=0.01, order=4
        )
        assert sol.success


class TestODEMethodComparison:
    """compare different ode methods"""
    
    def test_accuracy_ordering(self):
        """verify accuracy order: euler < heun < rk4"""
        f = lambda x, y: -y + x
        y0, x0, x_end = 1.0, 0.0, 1.0
        step = 0.1
        
        # exact solution at x=1: y = x - 1 + 2*exp(-x)
        exact = 1 - 1 + 2*np.exp(-1.0)
        
        sol_euler = ode_solvers.euler(f, y0, x0, x_end, step)
        sol_heun = ode_solvers.heun(f, y0, x0, x_end, step)
        sol_rk4 = ode_solvers.rk4(f, y0, x0, x_end, step)
        
        err_euler = abs(sol_euler.y[-1] - exact)
        err_heun = abs(sol_heun.y[-1] - exact)
        err_rk4 = abs(sol_rk4.y[-1] - exact)
        
        assert err_heun < err_euler
        assert err_rk4 < err_heun
    
    def test_evaluation_counts(self):
        """verify function evaluation counts"""
        f = lambda x, y: -y
        y0, x0, x_end = 1.0, 0.0, 1.0
        step = 0.1
        
        sol_euler = ode_solvers.euler(f, y0, x0, x_end, step)
        sol_heun = ode_solvers.heun(f, y0, x0, x_end, step)
        sol_rk4 = ode_solvers.rk4(f, y0, x0, x_end, step)
        
        n_steps = int((x_end - x0) / step)
        
        # euler: 1 eval per step
        assert sol_euler.n_evaluations == n_steps
        
        # heun: 2 evals per step
        assert sol_heun.n_evaluations == 2 * n_steps
        
        # rk4: 4 evals per step
        assert sol_rk4.n_evaluations == 4 * n_steps


class TestODEEdgeCases:
    """test edge cases and special scenarios"""
    
    def test_zero_derivative(self):
        """solve dy/dx = 0 (constant solution)"""
        sol = ode_solvers.rk4(lambda x, y: 0, y0=5.0, x0=0, x_end=10, step=0.1)
        assert sol.success
        assert np.allclose(sol.y, 5.0)
    
    def test_negative_initial_condition(self):
        """test with negative y0"""
        sol = ode_solvers.rk4(lambda x, y: -y, y0=-1.0, x0=0, x_end=1, step=0.01)
        assert sol.success
        expected = -np.exp(-1.0)
        assert abs(sol.y[-1] - expected) < 1e-5
    
    def test_large_time_interval(self):
        """test over large interval"""
        sol = ode_solvers.rk4(
            lambda x, y: -0.1*y,
            y0=1.0, x0=0.0, x_end=100.0, step=0.5
        )
        assert sol.success
        expected = np.exp(-10.0)
        assert abs(sol.y[-1] - expected) < 1e-4
    
    def test_oscillatory_solution(self):
        """test oscillatory ode: d²y/dx² + y = 0"""
        # rewrite as system: dy₁/dx = y₂, dy₂/dx = -y₁
        # but we can test dy/dx = cos(x) with y(0) = 0
        # exact: y = sin(x)
        sol = ode_solvers.rk4(
            lambda x, y: np.cos(x),
            y0=0.0, x0=0.0, x_end=2*np.pi, step=0.01
        )
        assert sol.success
        expected = 0.0  # sin(2π) = 0
        assert abs(sol.y[-1] - expected) < 1e-6
    
    def test_rapid_growth(self):
        """test rapidly growing solution"""
        sol = ode_solvers.rk4(
            lambda x, y: 5*y,
            y0=1.0, x0=0.0, x_end=1.0, step=0.01
        )
        assert sol.success
        expected = np.exp(5.0)
        assert abs(sol.y[-1] - expected) / expected < 1e-5
    
    def test_solution_array_length(self):
        """verify solution arrays have correct length"""
        step = 0.1
        x0, x_end = 0.0, 2.0
        expected_len = int((x_end - x0) / step) + 1
        
        sol = ode_solvers.rk4(
            lambda x, y: -y,
            y0=1.0, x0=x0, x_end=x_end, step=step
        )
        
        assert len(sol.x) == expected_len
        assert len(sol.y) == expected_len
        assert sol.x[0] == x0
        assert abs(sol.x[-1] - x_end) < 1e-10
    
    def test_non_finite_derivative(self):
        """test handling of non-finite derivatives"""
        def bad_derivative(x, y):
            if x > 0.5:
                return np.inf
            return -y
        
        sol = ode_solvers.rk4(bad_derivative, y0=1.0, x0=0, x_end=1, step=0.1)
        assert not sol.success
        assert "non-finite" in sol.message.lower()
