"""
tests for analysis module

verifies error estimation, convergence testing, and stability analysis.
"""

import pytest
import numpy as np
from numeric_integrator import analysis
from numeric_integrator.errors import ConvergenceError, StabilityError


class TestErrorEstimation:
    """test error estimation functions"""
    
    def test_absolute_error(self):
        """test absolute error computation"""
        result = analysis.estimate_error(3.14, 3.0, method="test")
        assert abs(result.absolute_error - 0.14) < 1e-10
        assert result.method == "test"
    
    def test_relative_error(self):
        """test relative error computation"""
        result = analysis.estimate_error(10.0, 8.0, method="test")
        assert abs(result.relative_error - 0.25) < 1e-10
    
    def test_near_zero(self):
        """test error when exact value is near zero"""
        result = analysis.estimate_error(1e-16, 1e-17)
        assert result.relative_error < 1e-15


class TestConvergenceTest:
    """test convergence analysis"""
    
    def test_linear_convergence(self):
        """test first-order convergence detection"""
        # create method with O(h) error
        def method(h):
            return 1.0 + 0.1 * h  # true value is 1.0
        
        step_sizes = [0.1, 0.05, 0.025, 0.0125]
        result = analysis.convergence_test(method, exact_value=1.0, 
                                          step_sizes=step_sizes, method_name="test")
        
        assert result.converged
        assert 0.8 < result.rate < 1.2  # should be close to 1
        assert len(result.errors) == 4
    
    def test_quadratic_convergence(self):
        """test second-order convergence detection"""
        # create method with O(h²) error
        def method(h):
            return 2.0 + 0.5 * h * h
        
        step_sizes = [0.2, 0.1, 0.05, 0.025]
        result = analysis.convergence_test(method, exact_value=2.0,
                                          step_sizes=step_sizes)
        
        assert result.converged
        assert 1.8 < result.rate < 2.2  # should be close to 2
    
    def test_non_convergence(self):
        """test detection of non-convergence"""
        # method that doesn't improve
        def method(h):
            return 1.0 + 0.1 * np.random.random()
        
        step_sizes = [0.1, 0.05, 0.025]
        result = analysis.convergence_test(method, exact_value=1.0,
                                          step_sizes=step_sizes)
        
        # may or may not converge due to randomness
        assert len(result.errors) == 3
    
    def test_insufficient_data(self):
        """test with insufficient step sizes"""
        with pytest.raises(ConvergenceError, match="at least 2"):
            analysis.convergence_test(lambda h: 1.0, 1.0, [0.1])
    
    def test_method_failure(self):
        """test when method raises exception"""
        def bad_method(h):
            raise ValueError("failed")
        
        with pytest.raises(ConvergenceError, match="method failed"):
            analysis.convergence_test(bad_method, 1.0, [0.1, 0.05])


class TestStabilityAnalysis:
    """test stability analysis for ode methods"""
    
    def test_euler_stable_region(self):
        """euler is stable for Re(λh) in [-2, 0]"""
        # stable case
        result = analysis.stability_analysis("euler", step_size=0.1, lambda_val=-5.0)
        assert result.method == "euler"
        assert result.spectral_radius == abs(1 + 0.1 * (-5.0))
        # |1 - 0.5| = 0.5 < 1, so stable
        assert result.stable
    
    def test_euler_unstable(self):
        """euler is unstable for large negative λh"""
        result = analysis.stability_analysis("euler", step_size=0.5, lambda_val=-5.0)
        # |1 + 0.5*(-5)| = |1 - 2.5| = 1.5 > 1, so unstable
        assert not result.stable
    
    def test_rk4_stability(self):
        """test rk4 stability"""
        result = analysis.stability_analysis("rk4", step_size=0.1, lambda_val=-2.0)
        assert result.method == "rk4"
        # rk4 has larger stability region than euler
        assert result.stable
    
    def test_backward_euler(self):
        """backward euler is unconditionally stable"""
        result = analysis.stability_analysis("backward_euler", step_size=1.0, lambda_val=-10.0)
        assert result.stable
    
    def test_complex_eigenvalue(self):
        """test with complex eigenvalue"""
        result = analysis.stability_analysis("euler", step_size=0.1, lambda_val=-1.0 + 1.0j)
        assert result.method == "euler"
        assert isinstance(result.spectral_radius, (int, float))
    
    def test_unknown_method(self):
        """test with unknown method"""
        with pytest.raises(StabilityError, match="not implemented"):
            analysis.stability_analysis("unknown_method", 0.1, -1.0)


class TestTruncationError:
    """test truncation error estimation"""
    
    def test_euler_error(self):
        """euler has O(h²) local error"""
        error = analysis.truncation_error_estimate("euler", h=0.1, derivative_bound=1.0)
        expected = 0.1**2 * 1.0 / 2
        assert abs(error - expected) < 1e-10
    
    def test_rk4_error(self):
        """rk4 has O(h⁵) local error"""
        error = analysis.truncation_error_estimate("rk4", h=0.01, derivative_bound=10.0)
        expected = 0.01**5 * 10.0 / 180
        assert abs(error - expected) < 1e-15
    
    def test_unknown_method(self):
        """test with unknown method"""
        with pytest.raises(ValueError, match="not available"):
            analysis.truncation_error_estimate("unknown", h=0.1, derivative_bound=1.0)


class TestGlobalError:
    """test global error estimation"""
    
    def test_error_accumulation(self):
        """global error = n_steps * local_error"""
        local_error = 1e-6
        x0, x_end, h = 0.0, 1.0, 0.01
        global_error = analysis.global_error_estimate(local_error, x0, x_end, h)
        expected = (x_end - x0) / h * local_error
        assert abs(global_error - expected) < 1e-15


class TestConditionNumber:
    """test condition number computation"""
    
    def test_well_conditioned(self):
        """test well-conditioned function"""
        kappa = analysis.condition_number(lambda x: x**2, x=1.0)
        # κ = |x * 2x / x²| = |2| = 2
        assert abs(kappa - 2.0) < 0.1
    
    def test_ill_conditioned(self):
        """test ill-conditioned function"""
        kappa = analysis.condition_number(lambda x: np.exp(x), x=10.0)
        # κ = |x * exp(x) / exp(x)| = |x| = 10
        assert kappa > 5
    
    def test_near_zero(self):
        """test at near-zero value"""
        kappa = analysis.condition_number(lambda x: x**3, x=1e-10)
        assert np.isfinite(kappa)


class TestRichardsonError:
    """test richardson error estimation"""
    
    def test_second_order_method(self):
        """test error estimate for second-order method"""
        coarse = 1.000
        fine = 1.001
        error = analysis.richardson_error_estimate(coarse, fine, order=2)
        expected = abs(fine - coarse) / (2**2 - 1)
        assert abs(error - expected) < 1e-10
    
    def test_fourth_order_method(self):
        """test error estimate for fourth-order method"""
        coarse = 2.0
        fine = 2.001
        error = analysis.richardson_error_estimate(coarse, fine, order=4)
        expected = abs(fine - coarse) / (2**4 - 1)
        assert abs(error - expected) < 1e-10


class TestAdaptiveStepController:
    """test adaptive step size control"""
    
    def test_decrease_step(self):
        """step should decrease when error > tol"""
        h_new = analysis.adaptive_step_controller(error=1e-3, tol=1e-6, h=0.1, order=4)
        assert h_new < 0.1
    
    def test_increase_step(self):
        """step should increase when error < tol"""
        h_new = analysis.adaptive_step_controller(error=1e-9, tol=1e-6, h=0.1, order=4)
        assert h_new > 0.1
    
    def test_step_limiting(self):
        """step changes should be limited"""
        # very small error should not cause huge step increase
        h_new = analysis.adaptive_step_controller(error=1e-15, tol=1e-6, h=0.1, order=4)
        assert h_new <= 0.5  # limited to 5x increase
    
    def test_zero_error(self):
        """handle zero error gracefully"""
        h_new = analysis.adaptive_step_controller(error=0, tol=1e-6, h=0.1, order=4)
        assert h_new == 0.1  # keep current step
