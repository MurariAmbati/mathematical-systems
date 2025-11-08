#!/usr/bin/env python3
"""
demonstration script for symbolic solver.
showcases parsing, simplification, differentiation, integration, and evaluation.
"""

from symbolic import parse


def demo_parsing():
    """demonstrate expression parsing."""
    print("=== parsing ===\n")
    
    expressions = [
        "x^2 + 2*x + 1",
        "sin(x) * cos(x)",
        "(x + 1) / (x - 1)",
        "exp(x^2)"
    ]
    
    for expr_str in expressions:
        expr = parse(expr_str)
        print(f"parsed: {expr_str}")
        print(f"result: {expr}")
        print()


def demo_simplification():
    """demonstrate algebraic simplification."""
    print("=== simplification ===\n")
    
    cases = [
        "x + 0",
        "x * 1",
        "x * 0",
        "2 + 3",
        "(x + 0) * 1",
        "x^1",
        "--x"
    ]
    
    for expr_str in cases:
        expr = parse(expr_str)
        simplified = expr.simplify()
        print(f"{expr_str} = {simplified}")
    print()


def demo_differentiation():
    """demonstrate symbolic differentiation."""
    print("=== differentiation ===\n")
    
    cases = [
        "x^2",
        "x^3 + 2*x + 1",
        "sin(x)",
        "exp(x)",
        "x^2 * x^3",
        "1 / x",
        "sin(x^2)"
    ]
    
    for expr_str in cases:
        expr = parse(expr_str)
        derivative = expr.differentiate("x").simplify()
        print(f"d/dx({expr_str}) = {derivative}")
    print()


def demo_integration():
    """demonstrate symbolic integration."""
    print("=== integration ===\n")
    
    cases = [
        "x",
        "x^2",
        "x^2 + 2*x + 1",
        "sin(x)",
        "cos(x)",
        "exp(x)"
    ]
    
    for expr_str in cases:
        expr = parse(expr_str)
        integral = expr.integrate("x").simplify()
        print(f"∫{expr_str} dx = {integral}")
    print()


def demo_evaluation():
    """demonstrate numeric evaluation."""
    print("=== evaluation ===\n")
    
    expr = parse("x^2 + 2*x + 1")
    print(f"expression: {expr}")
    print(f"at x=0: {expr.evaluate(x=0.0)}")
    print(f"at x=1: {expr.evaluate(x=1.0)}")
    print(f"at x=3: {expr.evaluate(x=3.0)}")
    print()
    
    expr = parse("x^2 + y^2")
    print(f"expression: {expr}")
    print(f"at x=3, y=4: {expr.evaluate(x=3.0, y=4.0)}")
    print()


def demo_latex():
    """demonstrate latex output."""
    print("=== latex output ===\n")
    
    cases = [
        "x^2 + 2*x + 1",
        "x / (x + 1)",
        "sin(x) * cos(x)",
        "sqrt(x^2 + 1)"
    ]
    
    for expr_str in cases:
        expr = parse(expr_str)
        latex = expr.to_latex()
        print(f"expression: {expr_str}")
        print(f"latex: {latex}")
        print()


def demo_verification():
    """verify symbolic operations numerically."""
    print("=== numerical verification ===\n")
    
    # verify derivative
    expr = parse("x^3")
    derivative = expr.differentiate("x").simplify()
    
    x_val = 2.0
    symbolic_deriv = derivative.evaluate(x=x_val)
    
    # numerical derivative
    h = 0.0001
    f_plus = expr.evaluate(x=x_val + h)
    f_minus = expr.evaluate(x=x_val - h)
    numerical_deriv = (f_plus - f_minus) / (2 * h)
    
    print(f"function: x^3 at x={x_val}")
    print(f"symbolic derivative: {symbolic_deriv}")
    print(f"numerical derivative: {numerical_deriv}")
    print(f"difference: {abs(symbolic_deriv - numerical_deriv):.10f}")
    print()
    
    # verify integral
    expr = parse("x^2")
    integral = expr.integrate("x").simplify()
    recovered = integral.differentiate("x").simplify()
    
    print(f"function: x^2")
    print(f"integral: {integral}")
    print(f"derivative of integral: {recovered}")
    print(f"verification at x=3: {expr.evaluate(x=3.0)} ≈ {recovered.evaluate(x=3.0)}")
    print()


if __name__ == "__main__":
    print("symbolic solver demonstration")
    print("=" * 50)
    print()
    
    demo_parsing()
    demo_simplification()
    demo_differentiation()
    demo_integration()
    demo_evaluation()
    demo_latex()
    demo_verification()
    
    print("demonstration complete")
