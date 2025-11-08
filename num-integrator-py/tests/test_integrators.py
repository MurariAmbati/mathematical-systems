"""
tests for integration methods

verifies accuracy, error estimation, and edge cases for all integrators.
"""

import pytest
import numpy as np
from numeric_integrator import integrators
from numeric_integrator.errors import IntegrationError, ConvergenceError


class TestTrapezoidal:
    """test trapezoidal rule"""
    
    def test_polynomial(self):
        """integrate x² from 0 to 1 (exact = 1/3)"""
        result = integrators.trapezoidal(lambda x: x**2, 0, 1, n=100)
        assert abs(result.value - 1/3) < 1e-4
        assert result.method == "trapezoidal"
        assert result.n_evaluations == 101
    
    def test_sine(self):
        """integrate sin(x) from 0 to π (exact = 2)"""
        result = integrators.trapezoidal(np.sin, 0, np.pi, n=200)
        assert abs(result.value - 2.0) < 1e-4
    
    def test_exponential(self):
        """integrate exp(x) from 0 to 1 (exact = e - 1)"""
        result = integrators.trapezoidal(np.exp, 0, 1, n=150)
        expected = np.exp(1) - 1
        assert abs(result.value - expected) < 1e-3
    
    def test_invalid_n(self):
        """test with invalid number of intervals"""
        with pytest.raises(IntegrationError, match="must be positive"):
            integrators.trapezoidal(lambda x: x, 0, 1, n=0)
        with pytest.raises(IntegrationError, match="must be positive"):
            integrators.trapezoidal(lambda x: x, 0, 1, n=-10)
    
    def test_invalid_bounds(self):
        """test with invalid integration bounds"""
        with pytest.raises(IntegrationError, match="must be finite"):
            integrators.trapezoidal(lambda x: x, np.inf, 1, n=10)
        with pytest.raises(IntegrationError, match="less than upper bound"):
            integrators.trapezoidal(lambda x: x, 1, 0, n=10)
    
    def test_function_error(self):
        """test with function that raises error"""
        def bad_func(x):
            if x > 0.5:
                raise ValueError("invalid x")
            return x
        
        with pytest.raises(IntegrationError, match="function evaluation failed"):
            integrators.trapezoidal(bad_func, 0, 1, n=10)
    
    def test_nan_result(self):
        """test with function returning nan"""
        with pytest.raises(IntegrationError, match="non-finite"):
            integrators.trapezoidal(lambda x: np.nan, 0, 1, n=10)


class TestSimpson:
    """test simpson's rule"""
    
    def test_polynomial(self):
        """integrate x³ from 0 to 2 (exact = 4)"""
        result = integrators.simpson(lambda x: x**3, 0, 2, n=100)
        assert abs(result.value - 4.0) < 1e-8
        assert result.method == "simpson"
    
    def test_cosine(self):
        """integrate cos(x) from 0 to π/2 (exact = 1)"""
        result = integrators.simpson(np.cos, 0, np.pi/2, n=100)
        assert abs(result.value - 1.0) < 1e-10
    
    def test_odd_n(self):
        """test that odd n raises error"""
        with pytest.raises(IntegrationError, match="even number"):
            integrators.simpson(lambda x: x, 0, 1, n=101)
    
    def test_small_n(self):
        """test with small n"""
        result = integrators.simpson(lambda x: x**2, 0, 1, n=2)
        assert abs(result.value - 1/3) < 1e-2
    
    def test_quartic(self):
        """integrate x⁴ from -1 to 1 (exact = 2/5)"""
        result = integrators.simpson(lambda x: x**4, -1, 1, n=50)
        assert abs(result.value - 0.4) < 1e-10


class TestMidpoint:
    """test midpoint rule"""
    
    def test_linear(self):
        """integrate 2x + 3 from 0 to 5"""
        result = integrators.midpoint(lambda x: 2*x + 3, 0, 5, n=100)
        expected = 25 + 15  # integral = x² + 3x from 0 to 5
        assert abs(result.value - expected) < 1e-3
    
    def test_sqrt(self):
        """integrate sqrt(x) from 0 to 4 (exact = 16/3)"""
        result = integrators.midpoint(lambda x: np.sqrt(x), 0, 4, n=200)
        assert abs(result.value - 16/3) < 1e-2
    
    def test_negative_bounds(self):
        """test with negative bounds"""
        result = integrators.midpoint(lambda x: x**2, -2, 2, n=100)
        expected = 16/3  # symmetric integral
        assert abs(result.value - expected) < 1e-3


class TestBoole:
    """test boole's rule"""
    
    def test_polynomial_fifth(self):
        """integrate x⁵ from 0 to 1 (exact = 1/6)"""
        result = integrators.boole(lambda x: x**5, 0, 1, n=100)
        assert abs(result.value - 1/6) < 1e-12
    
    def test_n_not_divisible_by_4(self):
        """test that n not divisible by 4 raises error"""
        with pytest.raises(IntegrationError, match="divisible by 4"):
            integrators.boole(lambda x: x, 0, 1, n=10)
    
    def test_sin_squared(self):
        """integrate sin²(x) from 0 to π (exact = π/2)"""
        result = integrators.boole(lambda x: np.sin(x)**2, 0, np.pi, n=100)
        assert abs(result.value - np.pi/2) < 1e-10


class TestRomberg:
    """test romberg integration"""
    
    def test_polynomial(self):
        """integrate x² from 0 to 1"""
        result = integrators.romberg(lambda x: x**2, 0, 1, max_iter=10, tol=1e-10)
        assert abs(result.value - 1/3) < 1e-10
        assert result.method == "romberg"
    
    def test_exponential(self):
        """integrate exp(-x²) from -2 to 2"""
        result = integrators.romberg(lambda x: np.exp(-x**2), -2, 2, max_iter=15, tol=1e-12)
        expected = np.sqrt(np.pi) * 0.9953222650  # approximate
        assert abs(result.value - expected) < 1e-8
    
    def test_convergence_failure(self):
        """test non-convergence with too few iterations"""
        # use oscillatory function that's hard to integrate
        with pytest.raises(ConvergenceError, match="did not converge"):
            integrators.romberg(lambda x: np.sin(100*x), 0, 1, max_iter=2, tol=1e-10)
    
    def test_invalid_max_iter(self):
        """test with invalid max_iter"""
        with pytest.raises(IntegrationError, match="must be positive"):
            integrators.romberg(lambda x: x, 0, 1, max_iter=0)


class TestAdaptiveTrapezoidal:
    """test adaptive trapezoidal rule"""
    
    def test_smooth_function(self):
        """integrate smooth function"""
        result = integrators.adaptive_trapezoidal(lambda x: x**3, 0, 1, tol=1e-8)
        assert abs(result.value - 0.25) < 1e-7
        assert result.method == "adaptive_trapezoidal"
    
    def test_discontinuous_derivative(self):
        """integrate function with discontinuous derivative"""
        def f(x):
            return abs(x - 0.5)
        
        result = integrators.adaptive_trapezoidal(f, 0, 1, tol=1e-6)
        expected = 0.25  # triangle area
        assert abs(result.value - expected) < 1e-5
    
    def test_max_depth_exceeded(self):
        """test maximum recursion depth"""
        # pathological function requiring deep recursion
        with pytest.raises(IntegrationError, match="maximum recursion depth"):
            integrators.adaptive_trapezoidal(
                lambda x: np.sin(1/(x + 0.01)), 0, 1, tol=1e-12, max_depth=5
            )
    
    def test_oscillatory(self):
        """integrate oscillatory function"""
        result = integrators.adaptive_trapezoidal(
            lambda x: np.sin(20*x), 0, 2*np.pi, tol=1e-6
        )
        assert abs(result.value) < 1e-5  # integral should be near zero


class TestAdaptiveSimpson:
    """test adaptive simpson's rule"""
    
    def test_polynomial(self):
        """integrate high-degree polynomial"""
        result = integrators.adaptive_simpson(lambda x: x**7, 0, 1, tol=1e-10)
        assert abs(result.value - 1/8) < 1e-10
        assert result.method == "adaptive_simpson"
    
    def test_rational_function(self):
        """integrate 1/(1+x²) from 0 to 1 (exact = π/4)"""
        result = integrators.adaptive_simpson(
            lambda x: 1/(1 + x**2), 0, 1, tol=1e-10
        )
        assert abs(result.value - np.pi/4) < 1e-10
    
    def test_gaussian(self):
        """integrate gaussian exp(-x²/2) from -3 to 3"""
        result = integrators.adaptive_simpson(
            lambda x: np.exp(-x**2/2), -3, 3, tol=1e-8
        )
        expected = np.sqrt(2*np.pi) * 0.9973  # approximate
        assert abs(result.value - expected) < 1e-3
    
    def test_singularity_handling(self):
        """test function with near-singularity"""
        result = integrators.adaptive_simpson(
            lambda x: 1/np.sqrt(x + 0.01), 0, 1, tol=1e-6
        )
        expected = 2 * (np.sqrt(1.01) - np.sqrt(0.01))
        assert abs(result.value - expected) < 1e-4
    
    def test_high_accuracy(self):
        """test high accuracy requirement"""
        result = integrators.adaptive_simpson(
            lambda x: np.cos(x), 0, np.pi/2, tol=1e-12
        )
        assert abs(result.value - 1.0) < 1e-12


class TestIntegrationEdgeCases:
    """test edge cases and special scenarios"""
    
    def test_constant_function(self):
        """integrate constant function"""
        result = integrators.simpson(lambda x: 5.0, 0, 10, n=10)
        assert abs(result.value - 50.0) < 1e-10
    
    def test_zero_function(self):
        """integrate zero function"""
        result = integrators.trapezoidal(lambda x: 0.0, 0, 1, n=10)
        assert abs(result.value) < 1e-15
    
    def test_negative_function(self):
        """integrate negative function"""
        result = integrators.simpson(lambda x: -x**2, 0, 1, n=100)
        assert abs(result.value + 1/3) < 1e-8
    
    def test_very_small_interval(self):
        """integrate over very small interval"""
        result = integrators.trapezoidal(lambda x: x**2, 0, 1e-6, n=10)
        expected = (1e-6)**3 / 3
        assert abs(result.value - expected) < 1e-18
    
    def test_large_values(self):
        """integrate function with large values"""
        result = integrators.simpson(lambda x: 1e10 * x, 0, 1, n=100)
        expected = 5e9
        assert abs(result.value - expected) / expected < 1e-8
