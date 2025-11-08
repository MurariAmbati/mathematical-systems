"""Tests for the printer module."""

import pytest
from symbolic.parser import parse


class TestToString:
    """Test string representation of expressions."""
    
    def test_constant_to_string(self) -> None:
        expr = parse("42")
        assert expr.to_string() == "42"
    
    def test_variable_to_string(self) -> None:
        expr = parse("x")
        assert expr.to_string() == "x"
    
    def test_addition_to_string(self) -> None:
        expr = parse("x + 1")
        result = expr.to_string()
        assert "x" in result
        assert "+" in result
        assert "1" in result
    
    def test_multiplication_to_string(self) -> None:
        expr = parse("2 * x")
        result = expr.to_string()
        assert "2" in result
        assert "*" in result
        assert "x" in result
    
    def test_power_to_string(self) -> None:
        expr = parse("x ^ 2")
        result = expr.to_string()
        assert "x" in result
        assert "^" in result
        assert "2" in result


class TestParentheses:
    """Test parentheses in string output."""
    
    def test_addition_in_multiplication(self) -> None:
        expr = parse("(x + 1) * 2")
        result = expr.to_string()
        # Should have parentheses around x + 1
        assert "(" in result and ")" in result
    
    def test_multiplication_in_addition(self) -> None:
        expr = parse("2 * x + 1")
        result = expr.to_string()
        # No extra parentheses needed
        assert result.count("(") == 0
    
    def test_nested_parentheses(self) -> None:
        expr = parse("((x + 1) * 2) + 3")
        result = expr.to_string()
        assert "(" in result


class TestFunctionToString:
    """Test function string representation."""
    
    def test_sin_to_string(self) -> None:
        expr = parse("sin(x)")
        result = expr.to_string()
        assert "sin" in result
        assert "x" in result
    
    def test_nested_function_to_string(self) -> None:
        expr = parse("sin(cos(x))")
        result = expr.to_string()
        assert "sin" in result
        assert "cos" in result


class TestUnaryToString:
    """Test unary operator string representation."""
    
    def test_negation_to_string(self) -> None:
        expr = parse("-x")
        result = expr.to_string()
        assert result == "-x"
    
    def test_negation_with_parentheses(self) -> None:
        expr = parse("-(x + 1)")
        result = expr.to_string()
        assert "(" in result


class TestToLatex:
    """Test LaTeX representation of expressions."""
    
    def test_constant_to_latex(self) -> None:
        expr = parse("42")
        assert expr.to_latex() == "42"
    
    def test_variable_to_latex(self) -> None:
        expr = parse("x")
        assert expr.to_latex() == "x"
    
    def test_power_to_latex(self) -> None:
        expr = parse("x ^ 2")
        result = expr.to_latex()
        # Should be x^{2}
        assert "^{" in result and "}" in result
    
    def test_division_to_latex(self) -> None:
        expr = parse("x / 2")
        result = expr.to_latex()
        # Should use \frac
        assert "\\frac" in result
    
    def test_multiplication_to_latex(self) -> None:
        expr = parse("2 * x")
        result = expr.to_latex()
        # Should use implicit multiplication or \cdot
        assert "2" in result and "x" in result


class TestFunctionToLatex:
    """Test LaTeX representation of functions."""
    
    def test_sin_to_latex(self) -> None:
        expr = parse("sin(x)")
        result = expr.to_latex()
        assert "\\sin" in result
    
    def test_cos_to_latex(self) -> None:
        expr = parse("cos(x)")
        result = expr.to_latex()
        assert "\\cos" in result
    
    def test_sqrt_to_latex(self) -> None:
        expr = parse("sqrt(x)")
        result = expr.to_latex()
        assert "\\sqrt" in result


class TestComplexToLatex:
    """Test LaTeX for complex expressions."""
    
    def test_fraction_with_polynomial(self) -> None:
        expr = parse("(x + 1) / (x - 1)")
        result = expr.to_latex()
        assert "\\frac" in result
    
    def test_nested_power(self) -> None:
        expr = parse("x ^ (y + 1)")
        result = expr.to_latex()
        assert "^{" in result


class TestPiAndE:
    """Test special constants in output."""
    
    def test_pi_to_string(self) -> None:
        expr = parse("pi")
        result = expr.to_string()
        assert "π" in result or "pi" in result
    
    def test_pi_to_latex(self) -> None:
        expr = parse("pi")
        result = expr.to_latex()
        assert "\\pi" in result or "pi" in result
    
    def test_e_to_string(self) -> None:
        expr = parse("e")
        result = expr.to_string()
        assert "e" in result


class TestSimplifiedToString:
    """Test string output after simplification."""
    
    def test_simplified_addition(self) -> None:
        expr = parse("x + 0")
        simplified = expr.simplify()
        result = simplified.to_string()
        assert result == "x"
    
    def test_simplified_multiplication(self) -> None:
        expr = parse("x * 1")
        simplified = expr.simplify()
        result = simplified.to_string()
        assert result == "x"


class TestComplexExpressions:
    """Test string output for complex expressions."""
    
    def test_polynomial_to_string(self) -> None:
        expr = parse("x^2 + 2*x + 1")
        result = expr.to_string()
        assert all(s in result for s in ["x", "2", "1", "+"])
    
    def test_rational_function_to_string(self) -> None:
        expr = parse("(x^2 + 1) / (x + 1)")
        result = expr.to_string()
        assert "/" in result
    
    def test_trig_expression_to_string(self) -> None:
        expr = parse("sin(x) * cos(x)")
        result = expr.to_string()
        assert "sin" in result and "cos" in result
