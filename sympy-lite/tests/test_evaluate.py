"""Tests for the evaluation module (testing evaluate method on expressions)."""

import pytest
import math
from symbolic.parser import parse
from symbolic.errors import EvaluationError


class TestBasicEvaluation:
    """Test basic expression evaluation."""
    
    def test_constant_evaluation(self) -> None:
        expr = parse("42")
        result = expr.evaluate()
        assert result == 42.0
    
    def test_variable_evaluation(self) -> None:
        expr = parse("x")
        result = expr.evaluate(x=10.0)
        assert result == 10.0
    
    def test_pi_evaluation(self) -> None:
        expr = parse("pi")
        result = expr.evaluate()
        assert abs(result - math.pi) < 0.001
    
    def test_e_evaluation(self) -> None:
        expr = parse("e")
        result = expr.evaluate()
        assert abs(result - math.e) < 0.001


class TestArithmeticEvaluation:
    """Test arithmetic operation evaluation."""
    
    def test_addition(self) -> None:
        expr = parse("3 + 4")
        result = expr.evaluate()
        assert result == 7.0
    
    def test_subtraction(self) -> None:
        expr = parse("10 - 3")
        result = expr.evaluate()
        assert result == 7.0
    
    def test_multiplication(self) -> None:
        expr = parse("5 * 6")
        result = expr.evaluate()
        assert result == 30.0
    
    def test_division(self) -> None:
        expr = parse("15 / 3")
        result = expr.evaluate()
        assert result == 5.0
    
    def test_power(self) -> None:
        expr = parse("2 ^ 8")
        result = expr.evaluate()
        assert result == 256.0


class TestVariableSubstitution:
    """Test variable substitution."""
    
    def test_single_variable(self) -> None:
        expr = parse("x^2 + 2*x + 1")
        result = expr.evaluate(x=3.0)
        # 9 + 6 + 1 = 16
        assert result == 16.0
    
    def test_multiple_variables(self) -> None:
        expr = parse("x^2 + y^2")
        result = expr.evaluate(x=3.0, y=4.0)
        # 9 + 16 = 25
        assert result == 25.0
    
    def test_variable_in_function(self) -> None:
        expr = parse("sin(x)")
        result = expr.evaluate(x=0.0)
        assert abs(result - 0.0) < 0.001


class TestFunctionEvaluation:
    """Test mathematical function evaluation."""
    
    def test_sin(self) -> None:
        expr = parse("sin(0)")
        result = expr.evaluate()
        assert abs(result - 0.0) < 0.001
    
    def test_cos(self) -> None:
        expr = parse("cos(0)")
        result = expr.evaluate()
        assert abs(result - 1.0) < 0.001
    
    def test_exp(self) -> None:
        expr = parse("exp(0)")
        result = expr.evaluate()
        assert abs(result - 1.0) < 0.001
    
    def test_log(self) -> None:
        expr = parse("log(e)")
        result = expr.evaluate()
        assert abs(result - 1.0) < 0.001
    
    def test_sqrt(self) -> None:
        expr = parse("sqrt(16)")
        result = expr.evaluate()
        assert abs(result - 4.0) < 0.001


class TestComplexEvaluation:
    """Test evaluation of complex expressions."""
    
    def test_polynomial(self) -> None:
        expr = parse("x^3 - 2*x^2 + x - 1")
        result = expr.evaluate(x=2.0)
        # 8 - 8 + 2 - 1 = 1
        assert abs(result - 1.0) < 0.001
    
    def test_nested_functions(self) -> None:
        expr = parse("sin(cos(0))")
        result = expr.evaluate()
        # sin(cos(0)) = sin(1)
        expected = math.sin(1.0)
        assert abs(result - expected) < 0.001
    
    def test_mixed_expression(self) -> None:
        expr = parse("exp(x) + sin(x)")
        result = expr.evaluate(x=0.0)
        # exp(0) + sin(0) = 1 + 0 = 1
        assert abs(result - 1.0) < 0.001


class TestUnaryOperators:
    """Test unary operator evaluation."""
    
    def test_negation(self) -> None:
        expr = parse("-5")
        result = expr.evaluate()
        assert result == -5.0
    
    def test_negation_variable(self) -> None:
        expr = parse("-x")
        result = expr.evaluate(x=10.0)
        assert result == -10.0
    
    def test_double_negation(self) -> None:
        expr = parse("--5")
        result = expr.evaluate()
        assert result == 5.0


class TestErrorCases:
    """Test error handling in evaluation."""
    
    def test_missing_variable(self) -> None:
        expr = parse("x + y")
        with pytest.raises(EvaluationError):
            expr.evaluate(x=1.0)  # y is missing
    
    def test_division_by_zero(self) -> None:
        expr = parse("1 / 0")
        with pytest.raises(EvaluationError):
            expr.evaluate()
    
    def test_log_of_negative(self) -> None:
        expr = parse("log(-1)")
        with pytest.raises(EvaluationError):
            expr.evaluate()
    
    def test_sqrt_of_negative(self) -> None:
        expr = parse("sqrt(-1)")
        with pytest.raises(EvaluationError):
            expr.evaluate()


class TestEvaluateAfterOperations:
    """Test evaluation after symbolic operations."""
    
    def test_evaluate_after_simplify(self) -> None:
        expr = parse("x + 0")
        simplified = expr.simplify()
        result = simplified.evaluate(x=5.0)
        assert result == 5.0
    
    def test_evaluate_after_differentiate(self) -> None:
        expr = parse("x^2")
        derivative = expr.differentiate("x")
        result = derivative.evaluate(x=3.0)
        # d(x^2)/dx at x=3 is 2*3 = 6
        assert abs(result - 6.0) < 0.001
    
    def test_evaluate_after_integrate(self) -> None:
        expr = parse("x")
        integral = expr.integrate("x")
        result = integral.evaluate(x=2.0)
        # ∫x dx = x^2/2, at x=2 is 2
        assert abs(result - 2.0) < 0.001


class TestFloatingPointAccuracy:
    """Test floating point accuracy in evaluation."""
    
    def test_small_numbers(self) -> None:
        expr = parse("x^2")
        result = expr.evaluate(x=0.0001)
        assert abs(result - 0.00000001) < 1e-10
    
    def test_large_numbers(self) -> None:
        expr = parse("x^2")
        result = expr.evaluate(x=1000.0)
        assert abs(result - 1000000.0) < 0.001
    
    def test_negative_numbers(self) -> None:
        expr = parse("x^2 + x")
        result = expr.evaluate(x=-5.0)
        # 25 + (-5) = 20
        assert abs(result - 20.0) < 0.001
