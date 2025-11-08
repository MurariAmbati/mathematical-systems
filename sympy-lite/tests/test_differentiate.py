"""Tests for the differentiation module."""

import pytest
import math
from symbolic.parser import parse
from symbolic.ast import Constant


class TestBasicDerivatives:
    """Test basic differentiation rules."""
    
    def test_constant_derivative(self) -> None:
        # d(5)/dx = 0
        expr = parse("5")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 0.0
    
    def test_variable_derivative_same(self) -> None:
        # d(x)/dx = 1
        expr = parse("x")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 1.0
    
    def test_variable_derivative_different(self) -> None:
        # d(y)/dx = 0
        expr = parse("y")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 0.0


class TestLinearDerivatives:
    """Test derivatives of linear expressions."""
    
    def test_sum_rule(self) -> None:
        # d(x + y)/dx = 1 + 0 = 1
        expr = parse("x + y")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 1.0
    
    def test_difference_rule(self) -> None:
        # d(x - 5)/dx = 1 - 0 = 1
        expr = parse("x - 5")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 1.0
    
    def test_constant_multiple(self) -> None:
        # d(3*x)/dx = 3
        expr = parse("3 * x")
        result = expr.differentiate("x")
        result = result.simplify()
        assert isinstance(result, Constant)
        assert result.value == 3.0


class TestPowerRule:
    """Test power rule derivatives."""
    
    def test_x_squared(self) -> None:
        # d(x^2)/dx = 2*x
        expr = parse("x ^ 2")
        result = expr.differentiate("x")
        result = result.simplify()
        # Check by evaluating at x=3: should be 6
        value = result.evaluate(x=3.0)
        assert abs(value - 6.0) < 0.001
    
    def test_x_cubed(self) -> None:
        # d(x^3)/dx = 3*x^2
        expr = parse("x ^ 3")
        result = expr.differentiate("x")
        result = result.simplify()
        # Check by evaluating at x=2: should be 12
        value = result.evaluate(x=2.0)
        assert abs(value - 12.0) < 0.001
    
    def test_square_root(self) -> None:
        # d(x^0.5)/dx = 0.5*x^(-0.5)
        expr = parse("x ^ 0.5")
        result = expr.differentiate("x")
        result = result.simplify()
        # Check by evaluating at x=4: should be 0.25
        value = result.evaluate(x=4.0)
        assert abs(value - 0.25) < 0.001


class TestProductRule:
    """Test product rule."""
    
    def test_simple_product(self) -> None:
        # d(x * x)/dx = x + x = 2*x
        expr = parse("x * x")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=5: should be 10
        value = result.evaluate(x=5.0)
        assert abs(value - 10.0) < 0.001
    
    def test_polynomial_product(self) -> None:
        # d(x^2 * x^3)/dx = 2*x*x^3 + x^2*3*x^2 = 5*x^4
        expr = parse("x^2 * x^3")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=2: should be 5*16 = 80
        value = result.evaluate(x=2.0)
        assert abs(value - 80.0) < 0.001


class TestQuotientRule:
    """Test quotient rule."""
    
    def test_one_over_x(self) -> None:
        # d(1/x)/dx = -1/x^2
        expr = parse("1 / x")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=2: should be -0.25
        value = result.evaluate(x=2.0)
        assert abs(value - (-0.25)) < 0.001
    
    def test_x_over_x_squared(self) -> None:
        # d(x / x^2)/dx = d(1/x)/dx = -1/x^2
        expr = parse("x / (x^2)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=3: should be -1/9
        value = result.evaluate(x=3.0)
        assert abs(value - (-1.0/9.0)) < 0.001


class TestChainRule:
    """Test chain rule with functions."""
    
    def test_sin_of_x(self) -> None:
        # d(sin(x))/dx = cos(x)
        expr = parse("sin(x)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=0: should be cos(0) = 1
        value = result.evaluate(x=0.0)
        assert abs(value - 1.0) < 0.001
    
    def test_cos_of_x(self) -> None:
        # d(cos(x))/dx = -sin(x)
        expr = parse("cos(x)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=0: should be -sin(0) = 0
        value = result.evaluate(x=0.0)
        assert abs(value - 0.0) < 0.001
    
    def test_exp_of_x(self) -> None:
        # d(exp(x))/dx = exp(x)
        expr = parse("exp(x)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=0: should be exp(0) = 1
        value = result.evaluate(x=0.0)
        assert abs(value - 1.0) < 0.001
    
    def test_log_of_x(self) -> None:
        # d(log(x))/dx = 1/x
        expr = parse("log(x)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=2: should be 0.5
        value = result.evaluate(x=2.0)
        assert abs(value - 0.5) < 0.001
    
    def test_sin_of_x_squared(self) -> None:
        # d(sin(x^2))/dx = cos(x^2) * 2*x
        expr = parse("sin(x^2)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=0: should be 0
        value = result.evaluate(x=0.0)
        assert abs(value - 0.0) < 0.001


class TestPolynomialDerivatives:
    """Test derivatives of polynomials."""
    
    def test_quadratic(self) -> None:
        # d(x^2 + 2*x + 1)/dx = 2*x + 2
        expr = parse("x^2 + 2*x + 1")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=3: should be 2*3 + 2 = 8
        value = result.evaluate(x=3.0)
        assert abs(value - 8.0) < 0.001
    
    def test_cubic(self) -> None:
        # d(x^3 - 3*x^2 + 2*x)/dx = 3*x^2 - 6*x + 2
        expr = parse("x^3 - 3*x^2 + 2*x")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=2: should be 3*4 - 6*2 + 2 = 2
        value = result.evaluate(x=2.0)
        assert abs(value - 2.0) < 0.001


class TestComplexDerivatives:
    """Test derivatives of complex expressions."""
    
    def test_product_of_functions(self) -> None:
        # d(sin(x) * cos(x))/dx = cos(x)*cos(x) + sin(x)*(-sin(x))
        # = cos^2(x) - sin^2(x) = cos(2x)
        expr = parse("sin(x) * cos(x)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Evaluate at x=0: should be 1
        value = result.evaluate(x=0.0)
        assert abs(value - 1.0) < 0.001
    
    def test_quotient_of_polynomials(self) -> None:
        # d((x^2 + 1) / (x + 1))/dx
        expr = parse("(x^2 + 1) / (x + 1)")
        result = expr.differentiate("x")
        result = result.simplify()
        # Should have a value at x=1
        value = result.evaluate(x=1.0)
        # At x=1: (2*1*(1+1) - (1^2+1)*1) / (1+1)^2 = (4 - 2) / 4 = 0.5
        assert abs(value - 0.5) < 0.001


class TestNumericalVerification:
    """Verify derivatives numerically using finite differences."""
    
    def test_numerical_derivative_polynomial(self) -> None:
        # Test d(x^2)/dx numerically
        expr = parse("x^2")
        derivative = expr.differentiate("x").simplify()
        
        # Check at x=3
        x_val = 3.0
        symbolic_deriv = derivative.evaluate(x=x_val)
        
        # Numerical derivative using finite difference
        h = 0.0001
        f_plus = expr.evaluate(x=x_val + h)
        f_minus = expr.evaluate(x=x_val - h)
        numerical_deriv = (f_plus - f_minus) / (2 * h)
        
        assert abs(symbolic_deriv - numerical_deriv) < 0.01
    
    def test_numerical_derivative_trig(self) -> None:
        # Test d(sin(x))/dx numerically
        expr = parse("sin(x)")
        derivative = expr.differentiate("x").simplify()
        
        x_val = 1.0
        symbolic_deriv = derivative.evaluate(x=x_val)
        
        # Numerical derivative
        h = 0.0001
        f_plus = expr.evaluate(x=x_val + h)
        f_minus = expr.evaluate(x=x_val - h)
        numerical_deriv = (f_plus - f_minus) / (2 * h)
        
        assert abs(symbolic_deriv - numerical_deriv) < 0.001


class TestMultipleVariables:
    """Test differentiation with multiple variables."""
    
    def test_partial_derivative_x(self) -> None:
        # d(x*y)/dx = y
        expr = parse("x * y")
        result = expr.differentiate("x")
        result = result.simplify()
        # Should just be y
        value = result.evaluate(x=5.0, y=3.0)
        assert abs(value - 3.0) < 0.001
    
    def test_partial_derivative_y(self) -> None:
        # d(x*y)/dy = x
        expr = parse("x * y")
        result = expr.differentiate("y")
        result = result.simplify()
        # Should just be x
        value = result.evaluate(x=5.0, y=3.0)
        assert abs(value - 5.0) < 0.001
    
    def test_mixed_partial(self) -> None:
        # d(x^2 + y^2)/dx = 2*x
        expr = parse("x^2 + y^2")
        result = expr.differentiate("x")
        result = result.simplify()
        value = result.evaluate(x=3.0, y=4.0)
        assert abs(value - 6.0) < 0.001
