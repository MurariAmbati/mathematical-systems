"""Tests for the integration module."""

import pytest
import math
from symbolic.parser import parse
from symbolic.errors import IntegrationError


class TestBasicIntegration:
    """Test basic integration rules."""
    
    def test_constant_integration(self) -> None:
        # ∫5 dx = 5*x
        expr = parse("5")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=1.0) - 5.0) < 0.001
    
    def test_variable_integration(self) -> None:
        # ∫x dx = x^2/2
        expr = parse("x")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation: should get back x
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=3.0) - 3.0) < 0.001


class TestPowerRuleIntegration:
    """Test power rule for integration."""
    
    def test_x_squared(self) -> None:
        # ∫x^2 dx = x^3/3
        expr = parse("x^2")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify: d(x^3/3)/dx = x^2
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=2.0) - 4.0) < 0.001
    
    def test_x_cubed(self) -> None:
        # ∫x^3 dx = x^4/4
        expr = parse("x^3")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=2.0) - 8.0) < 0.001
    
    def test_x_to_half(self) -> None:
        # ∫x^0.5 dx = x^1.5/1.5
        expr = parse("x^0.5")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=4.0) - 2.0) < 0.001


class TestLinearityIntegration:
    """Test linearity of integration."""
    
    def test_sum_integration(self) -> None:
        # ∫(x + 1) dx = x^2/2 + x
        expr = parse("x + 1")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        expected = expr.simplify().evaluate(x=3.0)
        actual = derivative.evaluate(x=3.0)
        assert abs(actual - expected) < 0.001
    
    def test_difference_integration(self) -> None:
        # ∫(x^2 - x) dx = x^3/3 - x^2/2
        expr = parse("x^2 - x")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=2.0) - expr.evaluate(x=2.0)) < 0.001
    
    def test_constant_multiple(self) -> None:
        # ∫(3*x) dx = 3*x^2/2
        expr = parse("3 * x")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        assert abs(derivative.evaluate(x=5.0) - 15.0) < 0.001


class TestTrigonometricIntegration:
    """Test integration of trigonometric functions."""
    
    def test_sin_integration(self) -> None:
        # ∫sin(x) dx = -cos(x)
        expr = parse("sin(x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation: should get sin(x) back
        derivative = result.differentiate("x").simplify()
        x_val = 1.0
        original = expr.evaluate(x=x_val)
        recovered = derivative.evaluate(x=x_val)
        assert abs(original - recovered) < 0.001
    
    def test_cos_integration(self) -> None:
        # ∫cos(x) dx = sin(x)
        expr = parse("cos(x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 0.5
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001


class TestExponentialIntegration:
    """Test integration of exponential functions."""
    
    def test_exp_integration(self) -> None:
        # ∫exp(x) dx = exp(x)
        expr = parse("exp(x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 1.0
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001


class TestPolynomialIntegration:
    """Test integration of polynomials."""
    
    def test_quadratic(self) -> None:
        # ∫(x^2 + 2*x + 1) dx = x^3/3 + x^2 + x
        expr = parse("x^2 + 2*x + 1")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 2.0
        expected = expr.evaluate(x=x_val)
        actual = derivative.evaluate(x=x_val)
        assert abs(actual - expected) < 0.001
    
    def test_cubic_polynomial(self) -> None:
        # ∫(x^3 - 3*x^2 + 2*x) dx = x^4/4 - x^3 + x^2
        expr = parse("x^3 - 3*x^2 + 2*x")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 1.5
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001


class TestIntegrationVerification:
    """Verify that integration is the inverse of differentiation."""
    
    def test_integrate_then_differentiate_polynomial(self) -> None:
        # For any polynomial, ∫(df/dx)dx should give back f (up to constant)
        original = parse("x^3 + 2*x^2 - x + 5")
        derivative = original.differentiate("x")
        integrated = derivative.integrate("x")
        
        # The integrated result minus original should be constant
        # We can verify by checking that the derivative is close
        x_val = 2.0
        deriv_of_integrated = integrated.differentiate("x").simplify()
        deriv_of_original = original.differentiate("x").simplify()
        
        assert abs(deriv_of_integrated.evaluate(x=x_val) - deriv_of_original.evaluate(x=x_val)) < 0.001
    
    def test_integrate_then_differentiate_trig(self) -> None:
        # ∫(d/dx[sin(x)])dx = sin(x)
        original = parse("sin(x)")
        derivative = original.differentiate("x")
        integrated = derivative.integrate("x")
        
        # Differentiate again
        recovered = integrated.differentiate("x").simplify()
        
        x_val = 1.0
        assert abs(recovered.evaluate(x=x_val) - derivative.evaluate(x=x_val)) < 0.001


class TestLinearSubstitution:
    """Test integration with linear substitution."""
    
    def test_sin_of_ax(self) -> None:
        # ∫sin(2*x) dx = -cos(2*x)/2
        expr = parse("sin(2*x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 0.5
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001
    
    def test_cos_of_ax(self) -> None:
        # ∫cos(3*x) dx = sin(3*x)/3
        expr = parse("cos(3*x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 0.7
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001
    
    def test_exp_of_ax(self) -> None:
        # ∫exp(2*x) dx = exp(2*x)/2
        expr = parse("exp(2*x)")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 0.5
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001


class TestIntegrationErrors:
    """Test that unsupported integrations raise errors."""
    
    def test_product_of_variables(self) -> None:
        # ∫(x * sin(x)) dx - not supported without integration by parts
        expr = parse("x * sin(x)")
        with pytest.raises(IntegrationError):
            expr.integrate("x")
    
    def test_division_by_variable(self) -> None:
        # ∫(1 / (x^2 + 1)) dx - not supported
        expr = parse("1 / (x^2 + 1)")
        with pytest.raises(IntegrationError):
            expr.integrate("x")
    
    def test_complex_function_composition(self) -> None:
        # ∫sin(x^2) dx - requires special functions
        expr = parse("sin(x^2)")
        with pytest.raises(IntegrationError):
            expr.integrate("x")


class TestDivisionIntegration:
    """Test integration with division."""
    
    def test_constant_division(self) -> None:
        # ∫(x / 2) dx = x^2/4
        expr = parse("x / 2")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 4.0
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001
    
    def test_polynomial_division(self) -> None:
        # ∫(x^2 / 3) dx = x^3/9
        expr = parse("x^2 / 3")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        x_val = 3.0
        assert abs(derivative.evaluate(x=x_val) - expr.evaluate(x=x_val)) < 0.001


class TestMultipleVariables:
    """Test integration with multiple variables."""
    
    def test_integrate_constant_wrt_x(self) -> None:
        # ∫y dx = y*x
        expr = parse("y")
        result = expr.integrate("x")
        result = result.simplify()
        # Should be y*x
        value = result.evaluate(x=2.0, y=3.0)
        assert abs(value - 6.0) < 0.001
    
    def test_integrate_mixed(self) -> None:
        # ∫(x + y) dx = x^2/2 + y*x
        expr = parse("x + y")
        result = expr.integrate("x")
        result = result.simplify()
        # Verify by differentiation
        derivative = result.differentiate("x").simplify()
        val = derivative.evaluate(x=2.0, y=3.0)
        expected = expr.evaluate(x=2.0, y=3.0)
        assert abs(val - expected) < 0.001
