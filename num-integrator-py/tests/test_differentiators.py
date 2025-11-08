"""
tests for differentiation methods

verifies accuracy and error estimation for finite difference schemes.
"""

import pytest
import numpy as np
from numeric_integrator import differentiators
from numeric_integrator.errors import DifferentiationError


class TestForwardDifference:
    """test forward finite difference"""
    
    def test_linear(self):
        """derivative of 3x + 2 is 3"""
        result = differentiators.forward_difference(lambda x: 3*x + 2, x0=5.0, h=1e-5)
        assert abs(result.value - 3.0) < 1e-4
        assert result.method == "forward_difference"
        assert result.n_evaluations == 2
    
    def test_quadratic(self):
        """derivative of x² at x=3 is 6"""
        result = differentiators.forward_difference(lambda x: x**2, x0=3.0, h=1e-5)
        assert abs(result.value - 6.0) < 1e-4
    
    def test_sine(self):
        """derivative of sin(x) at x=0 is 1"""
        result = differentiators.forward_difference(np.sin, x0=0.0, h=1e-5)
        assert abs(result.value - 1.0) < 1e-4
    
    def test_exponential(self):
        """derivative of exp(x) at x=1 is e"""
        result = differentiators.forward_difference(np.exp, x0=1.0, h=1e-5)
        assert abs(result.value - np.e) < 1e-4
    
    def test_invalid_h(self):
        """test with invalid step size"""
        with pytest.raises(DifferentiationError, match="must be positive"):
            differentiators.forward_difference(lambda x: x, x0=0, h=0)
        with pytest.raises(DifferentiationError, match="must be positive"):
            differentiators.forward_difference(lambda x: x, x0=0, h=-1e-5)
    
    def test_invalid_x0(self):
        """test with invalid x0"""
        with pytest.raises(DifferentiationError, match="must be finite"):
            differentiators.forward_difference(lambda x: x, x0=np.inf, h=1e-5)
    
    def test_function_error(self):
        """test with function that raises error"""
        def bad_func(x):
            raise ValueError("error")
        
        with pytest.raises(DifferentiationError, match="function evaluation failed"):
            differentiators.forward_difference(bad_func, x0=0, h=1e-5)


class TestBackwardDifference:
    """test backward finite difference"""
    
    def test_cubic(self):
        """derivative of x³ at x=2 is 12"""
        result = differentiators.backward_difference(lambda x: x**3, x0=2.0, h=1e-5)
        assert abs(result.value - 12.0) < 1e-3
        assert result.method == "backward_difference"
    
    def test_log(self):
        """derivative of ln(x) at x=2 is 0.5"""
        result = differentiators.backward_difference(np.log, x0=2.0, h=1e-5)
        assert abs(result.value - 0.5) < 1e-4
    
    def test_cosine(self):
        """derivative of cos(x) at x=π/4"""
        result = differentiators.backward_difference(np.cos, x0=np.pi/4, h=1e-5)
        expected = -np.sin(np.pi/4)
        assert abs(result.value - expected) < 1e-4


class TestCentralDifference:
    """test central finite difference"""
    
    def test_polynomial(self):
        """derivative of 5x⁴ at x=1.5 is 67.5"""
        result = differentiators.central_difference(lambda x: 5*x**4, x0=1.5, h=1e-5)
        expected = 20 * 1.5**3
        assert abs(result.value - expected) < 1e-6
        assert result.method == "central_difference"
    
    def test_sine_at_pi_over_3(self):
        """derivative of sin(x) at x=π/3 is 0.5"""
        result = differentiators.central_difference(np.sin, x0=np.pi/3, h=1e-5)
        expected = np.cos(np.pi/3)
        assert abs(result.value - expected) < 1e-8
    
    def test_reciprocal(self):
        """derivative of 1/x at x=4 is -1/16"""
        result = differentiators.central_difference(lambda x: 1/x, x0=4.0, h=1e-5)
        assert abs(result.value + 1/16) < 1e-6
    
    def test_better_than_forward(self):
        """central should be more accurate than forward"""
        f = lambda x: np.exp(x**2)
        x0 = 1.0
        exact = 2 * x0 * np.exp(x0**2)
        
        forward = differentiators.forward_difference(f, x0, h=1e-4)
        central = differentiators.central_difference(f, x0, h=1e-4)
        
        assert abs(central.value - exact) < abs(forward.value - exact)


class TestRichardsonExtrapolation:
    """test richardson extrapolation"""
    
    def test_high_accuracy(self):
        """richardson should achieve very high accuracy"""
        result = differentiators.richardson_extrapolation(
            lambda x: np.sin(x), x0=0.0, h=1e-3, n_iter=4
        )
        assert abs(result.value - 1.0) < 1e-12
        assert result.method == "richardson_extrapolation"
    
    def test_polynomial(self):
        """derivative of x⁶ at x=2 is 192"""
        result = differentiators.richardson_extrapolation(
            lambda x: x**6, x0=2.0, h=1e-2, n_iter=5
        )
        expected = 6 * 2**5
        assert abs(result.value - expected) < 1e-10
    
    def test_tan(self):
        """derivative of tan(x) at x=π/6 is 4/3"""
        result = differentiators.richardson_extrapolation(
            np.tan, x0=np.pi/6, h=1e-3, n_iter=4
        )
        expected = 1 / np.cos(np.pi/6)**2
        assert abs(result.value - expected) < 1e-10
    
    def test_convergence(self):
        """test convergence with increasing iterations"""
        f = lambda x: np.exp(np.sin(x))
        x0 = 1.0
        
        errors = []
        for n in range(2, 6):
            result = differentiators.richardson_extrapolation(f, x0, h=1e-3, n_iter=n)
            # exact derivative: exp(sin(x)) * cos(x)
            exact = np.exp(np.sin(x0)) * np.cos(x0)
            errors.append(abs(result.value - exact))
        
        # errors should decrease
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i]
    
    def test_invalid_parameters(self):
        """test with invalid parameters"""
        with pytest.raises(DifferentiationError):
            differentiators.richardson_extrapolation(lambda x: x, x0=0, h=-1e-3)
        
        with pytest.raises(DifferentiationError):
            differentiators.richardson_extrapolation(lambda x: x, x0=0, h=1e-3, n_iter=0)


class TestSecondDerivative:
    """test second derivative computation"""
    
    def test_polynomial(self):
        """second derivative of x³ at x=2 is 12"""
        result = differentiators.second_derivative(lambda x: x**3, x0=2.0, h=1e-4)
        expected = 6 * 2  # d²/dx²(x³) = 6x
        assert abs(result.value - expected) < 1e-3
        assert result.method == "second_derivative"
        assert result.n_evaluations == 3
    
    def test_sine(self):
        """second derivative of sin(x) is -sin(x)"""
        x0 = np.pi/4
        result = differentiators.second_derivative(np.sin, x0=x0, h=1e-4)
        expected = -np.sin(x0)
        assert abs(result.value - expected) < 1e-6
    
    def test_exponential(self):
        """second derivative of exp(x) is exp(x)"""
        x0 = 1.5
        result = differentiators.second_derivative(np.exp, x0=x0, h=1e-4)
        expected = np.exp(x0)
        assert abs(result.value - expected) < 1e-5
    
    def test_quartic(self):
        """second derivative of x⁴ at x=3 is 108"""
        result = differentiators.second_derivative(lambda x: x**4, x0=3.0, h=1e-4)
        expected = 12 * 3**2
        assert abs(result.value - expected) < 1e-2


class TestDerivativeVector:
    """test gradient computation"""
    
    def test_linear_function(self):
        """gradient of 2x + 3y at (1, 2) is [2, 3]"""
        f = lambda v: 2*v[0] + 3*v[1]
        grad = differentiators.derivative_vector(f, np.array([1.0, 2.0]), h=1e-5)
        assert np.allclose(grad, [2.0, 3.0], atol=1e-4)
    
    def test_quadratic_function(self):
        """gradient of x² + y² at (3, 4) is [6, 8]"""
        f = lambda v: v[0]**2 + v[1]**2
        grad = differentiators.derivative_vector(f, np.array([3.0, 4.0]), h=1e-5)
        assert np.allclose(grad, [6.0, 8.0], atol=1e-3)
    
    def test_multidimensional(self):
        """gradient of x₁·x₂·x₃ at (2, 3, 4)"""
        f = lambda v: v[0] * v[1] * v[2]
        x = np.array([2.0, 3.0, 4.0])
        grad = differentiators.derivative_vector(f, x, h=1e-5)
        expected = [12.0, 8.0, 6.0]  # [x₂·x₃, x₁·x₃, x₁·x₂]
        assert np.allclose(grad, expected, atol=1e-3)
    
    def test_invalid_h(self):
        """test with invalid step size"""
        f = lambda v: v[0]**2
        with pytest.raises(DifferentiationError):
            differentiators.derivative_vector(f, np.array([1.0]), h=0)


class TestDifferentiationEdgeCases:
    """test edge cases and special scenarios"""
    
    def test_constant_function(self):
        """derivative of constant is zero"""
        result = differentiators.central_difference(lambda x: 42.0, x0=10.0, h=1e-5)
        assert abs(result.value) < 1e-10
    
    def test_negative_x(self):
        """test at negative x values"""
        result = differentiators.central_difference(lambda x: x**2, x0=-3.0, h=1e-5)
        assert abs(result.value + 6.0) < 1e-6
    
    def test_near_zero(self):
        """test near x=0"""
        result = differentiators.central_difference(lambda x: x**3, x0=1e-3, h=1e-6)
        expected = 3 * (1e-3)**2
        assert abs(result.value - expected) / expected < 0.1
    
    def test_large_derivative(self):
        """test function with large derivative"""
        result = differentiators.central_difference(lambda x: x**10, x0=2.0, h=1e-5)
        expected = 10 * 2**9
        assert abs(result.value - expected) / expected < 1e-6
    
    def test_oscillatory_function(self):
        """test rapidly oscillating function"""
        result = differentiators.central_difference(
            lambda x: np.sin(10*x), x0=1.0, h=1e-5
        )
        expected = 10 * np.cos(10 * 1.0)
        assert abs(result.value - expected) < 1e-4
    
    def test_different_step_sizes(self):
        """test that smaller h gives better accuracy"""
        f = lambda x: np.exp(x)
        x0 = 1.0
        exact = np.e
        
        result1 = differentiators.central_difference(f, x0, h=1e-3)
        result2 = differentiators.central_difference(f, x0, h=1e-5)
        result3 = differentiators.central_difference(f, x0, h=1e-7)
        
        err1 = abs(result1.value - exact)
        err2 = abs(result2.value - exact)
        err3 = abs(result3.value - exact)
        
        assert err2 < err1
        assert err3 < err2
