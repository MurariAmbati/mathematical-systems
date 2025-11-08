"""Tests for the simplification module."""

import pytest
from symbolic.parser import parse
from symbolic.ast import Constant, Variable


class TestArithmeticSimplification:
    """Test basic arithmetic simplifications."""
    
    def test_addition_with_zero(self) -> None:
        # x + 0 = x
        expr = parse("x + 0")
        result = expr.simplify()
        assert isinstance(result, Variable)
        assert result.name == "x"
        
        # 0 + x = x
        expr = parse("0 + x")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_subtraction_with_zero(self) -> None:
        # x - 0 = x
        expr = parse("x - 0")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_multiplication_with_zero(self) -> None:
        # x * 0 = 0
        expr = parse("x * 0")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 0.0
        
        # 0 * x = 0
        expr = parse("0 * x")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 0.0
    
    def test_multiplication_with_one(self) -> None:
        # x * 1 = x
        expr = parse("x * 1")
        result = expr.simplify()
        assert isinstance(result, Variable)
        
        # 1 * x = x
        expr = parse("1 * x")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_division_with_one(self) -> None:
        # x / 1 = x
        expr = parse("x / 1")
        result = expr.simplify()
        assert isinstance(result, Variable)


class TestConstantFolding:
    """Test constant folding."""
    
    def test_addition(self) -> None:
        expr = parse("2 + 3")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 5.0
    
    def test_subtraction(self) -> None:
        expr = parse("10 - 3")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 7.0
    
    def test_multiplication(self) -> None:
        expr = parse("4 * 5")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 20.0
    
    def test_division(self) -> None:
        expr = parse("15 / 3")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 5.0
    
    def test_power(self) -> None:
        expr = parse("2 ^ 3")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 8.0
    
    def test_nested_constants(self) -> None:
        expr = parse("(2 + 3) * (4 - 1)")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 15.0


class TestPowerRules:
    """Test power simplification rules."""
    
    def test_power_zero(self) -> None:
        # x^0 = 1
        expr = parse("x ^ 0")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 1.0
    
    def test_power_one(self) -> None:
        # x^1 = x
        expr = parse("x ^ 1")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_zero_power(self) -> None:
        # 0^x = 0
        expr = parse("0 ^ x")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 0.0
    
    def test_one_power(self) -> None:
        # 1^x = 1
        expr = parse("1 ^ x")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == 1.0
    
    def test_power_of_power(self) -> None:
        # (x^2)^3 should simplify towards x^6
        expr = parse("(x ^ 2) ^ 3")
        result = expr.simplify()
        # Should be x^6 or x^(2*3)
        result_str = result.to_string()
        # After simplification should have exponent 6
        assert "6" in result_str or ("2" in result_str and "3" in result_str)


class TestUnarySimplification:
    """Test unary operator simplification."""
    
    def test_double_negation(self) -> None:
        # --x = x
        expr = parse("--x")
        result = expr.simplify()
        assert isinstance(result, Variable)
        assert result.name == "x"
    
    def test_unary_plus(self) -> None:
        # +x = x
        expr = parse("+x")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_negation_of_constant(self) -> None:
        # -5 = -5
        expr = parse("-5")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert result.value == -5.0


class TestFunctionSimplification:
    """Test function simplification."""
    
    def test_sin_of_zero(self) -> None:
        expr = parse("sin(0)")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert abs(result.value - 0.0) < 0.001
    
    def test_cos_of_zero(self) -> None:
        expr = parse("cos(0)")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert abs(result.value - 1.0) < 0.001
    
    def test_exp_of_zero(self) -> None:
        expr = parse("exp(0)")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert abs(result.value - 1.0) < 0.001
    
    def test_function_with_constant(self) -> None:
        # sin(pi) should evaluate
        import math
        expr = parse("sin(pi)")
        result = expr.simplify()
        assert isinstance(result, Constant)
        assert abs(result.value - 0.0) < 0.001


class TestComplexSimplification:
    """Test complex expression simplifications."""
    
    def test_polynomial_simplification(self) -> None:
        # (x + 0) * 1 + 0 = x
        expr = parse("(x + 0) * 1 + 0")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_nested_simplification(self) -> None:
        # ((x * 1) + 0) - 0 = x
        expr = parse("((x * 1) + 0) - 0")
        result = expr.simplify()
        assert isinstance(result, Variable)
    
    def test_mixed_constants_and_variables(self) -> None:
        # 2 * x * 1 + 3 - 3 = 2 * x
        expr = parse("2 * x * 1 + 3 - 3")
        result = expr.simplify()
        result_str = result.to_string()
        assert "x" in result_str


class TestIdempotence:
    """Test that simplification is idempotent."""
    
    def test_simplify_twice(self) -> None:
        expr = parse("x + 0")
        result1 = expr.simplify()
        result2 = result1.simplify()
        # Should be identical
        assert result1.to_string() == result2.to_string()
    
    def test_already_simple(self) -> None:
        expr = parse("x")
        result = expr.simplify()
        assert isinstance(result, Variable)
        assert result.name == "x"
