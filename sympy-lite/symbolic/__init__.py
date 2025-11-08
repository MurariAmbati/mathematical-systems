"""
Symbolic Solver - A lightweight symbolic algebra engine.

This package provides tools for symbolic mathematics including:
- Expression parsing
- Algebraic simplification
- Symbolic differentiation
- Symbolic integration
- Numeric evaluation
- LaTeX rendering

Example:
    >>> from symbolic import parse
    >>> expr = parse("x^2 + 2*x + 1")
    >>> print(expr.differentiate("x"))
    2*x + 2
"""

from symbolic.ast import Expression, Constant, Variable, BinaryOp, UnaryOp, Function
from symbolic.parser import parse
from symbolic.errors import (
    SymbolicError,
    ParseError,
    EvaluationError,
    IntegrationError,
    SimplificationError,
)

__version__ = "0.1.0"
__all__ = [
    "Expression",
    "Constant",
    "Variable",
    "BinaryOp",
    "UnaryOp",
    "Function",
    "parse",
    "SymbolicError",
    "ParseError",
    "EvaluationError",
    "IntegrationError",
    "SimplificationError",
]
