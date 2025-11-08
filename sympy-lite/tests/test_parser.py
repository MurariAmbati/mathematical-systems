"""Tests for the parser module."""

import pytest
import math
from symbolic.parser import parse
from symbolic.ast import Constant, Variable, BinaryOp, UnaryOp, Function
from symbolic.errors import ParseError


class TestBasicParsing:
    """Test basic parsing functionality."""
    
    def test_parse_constant(self) -> None:
        expr = parse("42")
        assert isinstance(expr, Constant)
        assert expr.value == 42.0
    
    def test_parse_float(self) -> None:
        expr = parse("3.14")
        assert isinstance(expr, Constant)
        assert abs(expr.value - 3.14) < 0.001
    
    def test_parse_variable(self) -> None:
        expr = parse("x")
        assert isinstance(expr, Variable)
        assert expr.name == "x"
    
    def test_parse_special_constants(self) -> None:
        # Test pi
        expr = parse("pi")
        assert isinstance(expr, Constant)
        assert abs(expr.value - math.pi) < 0.001
        
        # Test e
        expr = parse("e")
        assert isinstance(expr, Constant)
        assert abs(expr.value - math.e) < 0.001


class TestBinaryOperations:
    """Test parsing of binary operations."""
    
    def test_parse_addition(self) -> None:
        expr = parse("x + 1")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, Variable)
        assert isinstance(expr.right, Constant)
    
    def test_parse_subtraction(self) -> None:
        expr = parse("x - 5")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "-"
    
    def test_parse_multiplication(self) -> None:
        expr = parse("2 * x")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
    
    def test_parse_division(self) -> None:
        expr = parse("x / 2")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "/"
    
    def test_parse_power(self) -> None:
        expr = parse("x ^ 2")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "^"


class TestOperatorPrecedence:
    """Test operator precedence."""
    
    def test_multiplication_before_addition(self) -> None:
        # 2 + 3 * 4 should parse as 2 + (3 * 4)
        expr = parse("2 + 3 * 4")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, Constant)
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "*"
    
    def test_power_before_multiplication(self) -> None:
        # 2 * 3 ^ 4 should parse as 2 * (3 ^ 4)
        expr = parse("2 * 3 ^ 4")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "^"
    
    def test_right_associative_power(self) -> None:
        # 2 ^ 3 ^ 4 should parse as 2 ^ (3 ^ 4)
        expr = parse("2 ^ 3 ^ 4")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "^"
        assert isinstance(expr.right, BinaryOp)
        assert expr.right.op == "^"


class TestParentheses:
    """Test parenthesized expressions."""
    
    def test_simple_parentheses(self) -> None:
        expr = parse("(x + 1)")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
    
    def test_precedence_override(self) -> None:
        # (2 + 3) * 4 should parse as (2 + 3) * 4
        expr = parse("(2 + 3) * 4")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        assert isinstance(expr.left, BinaryOp)
        assert expr.left.op == "+"
    
    def test_nested_parentheses(self) -> None:
        expr = parse("((x + 1) * 2)")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"


class TestUnaryOperators:
    """Test unary operators."""
    
    def test_negation(self) -> None:
        expr = parse("-x")
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"
        assert isinstance(expr.operand, Variable)
    
    def test_unary_plus(self) -> None:
        expr = parse("+x")
        assert isinstance(expr, UnaryOp)
        assert expr.op == "+"
    
    def test_double_negation(self) -> None:
        expr = parse("--x")
        assert isinstance(expr, UnaryOp)
        assert expr.op == "-"
        assert isinstance(expr.operand, UnaryOp)
        assert expr.operand.op == "-"
    
    def test_negation_with_operation(self) -> None:
        expr = parse("-x + 5")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "+"
        assert isinstance(expr.left, UnaryOp)


class TestFunctions:
    """Test function parsing."""
    
    def test_sin(self) -> None:
        expr = parse("sin(x)")
        assert isinstance(expr, Function)
        assert expr.name == "sin"
        assert isinstance(expr.arg, Variable)
    
    def test_cos(self) -> None:
        expr = parse("cos(x)")
        assert isinstance(expr, Function)
        assert expr.name == "cos"
    
    def test_exp(self) -> None:
        expr = parse("exp(x)")
        assert isinstance(expr, Function)
        assert expr.name == "exp"
    
    def test_log(self) -> None:
        expr = parse("log(x)")
        assert isinstance(expr, Function)
        assert expr.name == "log"
    
    def test_sqrt(self) -> None:
        expr = parse("sqrt(x)")
        assert isinstance(expr, Function)
        assert expr.name == "sqrt"
    
    def test_function_with_expression(self) -> None:
        expr = parse("sin(x + 1)")
        assert isinstance(expr, Function)
        assert isinstance(expr.arg, BinaryOp)
    
    def test_nested_functions(self) -> None:
        expr = parse("sin(cos(x))")
        assert isinstance(expr, Function)
        assert expr.name == "sin"
        assert isinstance(expr.arg, Function)
        assert expr.arg.name == "cos"


class TestComplexExpressions:
    """Test parsing of complex expressions."""
    
    def test_polynomial(self) -> None:
        expr = parse("x^2 + 2*x + 1")
        # Should be ((x^2) + (2*x)) + 1
        assert isinstance(expr, BinaryOp)
    
    def test_rational_function(self) -> None:
        expr = parse("(x + 1) / (x - 1)")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "/"
    
    def test_trigonometric_product(self) -> None:
        expr = parse("sin(x) * cos(x)")
        assert isinstance(expr, BinaryOp)
        assert expr.op == "*"
        assert isinstance(expr.left, Function)
        assert isinstance(expr.right, Function)
    
    def test_complex_nested(self) -> None:
        expr = parse("exp(sin(x^2 + 1))")
        assert isinstance(expr, Function)
        assert expr.name == "exp"
        assert isinstance(expr.arg, Function)


class TestWhitespace:
    """Test whitespace handling."""
    
    def test_whitespace_ignored(self) -> None:
        expr1 = parse("x+1")
        expr2 = parse("x + 1")
        expr3 = parse("  x   +   1  ")
        # All should parse to same structure
        assert expr1.to_string() == expr2.to_string() == expr3.to_string()


class TestErrorHandling:
    """Test error cases."""
    
    def test_empty_expression(self) -> None:
        with pytest.raises(ParseError):
            parse("")
    
    def test_mismatched_parentheses_open(self) -> None:
        with pytest.raises(ParseError):
            parse("(x + 1")
    
    def test_mismatched_parentheses_close(self) -> None:
        with pytest.raises(ParseError):
            parse("x + 1)")
    
    def test_invalid_character(self) -> None:
        with pytest.raises(ParseError):
            parse("x + @")
    
    def test_missing_operand(self) -> None:
        with pytest.raises(ParseError):
            parse("x +")
    
    def test_double_operator(self) -> None:
        # x ++ 1 is actually valid (x + (+1)), so test a truly invalid case
        with pytest.raises(ParseError):
            parse("x ** 1")  # ** is not supported, only ^
    
    def test_function_missing_parentheses(self) -> None:
        with pytest.raises(ParseError):
            parse("sin x")


class TestRoundTrip:
    """Test that parsed expressions can be converted back to strings."""
    
    def test_simple_roundtrip(self) -> None:
        original = "x + 1"
        expr = parse(original)
        result = expr.to_string()
        # Should be equivalent (may have different whitespace)
        assert "x" in result and "1" in result and "+" in result
    
    def test_complex_roundtrip(self) -> None:
        expr = parse("sin(x^2 + 1)")
        result = expr.to_string()
        assert "sin" in result and "x" in result and "2" in result
