"""
Expression rendering utilities for text and LaTeX output.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbolic.ast import BinaryOp, UnaryOp, Function, Expression


def needs_parentheses(expr: "Expression", parent_op: str, is_right: bool = False) -> bool:
    """Determine if an expression needs parentheses based on operator precedence."""
    from symbolic.ast import BinaryOp, UnaryOp, Constant, Variable, Function
    
    if isinstance(expr, (Constant, Variable, Function)):
        return False
    
    if isinstance(expr, UnaryOp):
        return False
    
    if isinstance(expr, BinaryOp):
        # Define precedence levels
        precedence = {
            "^": 4,
            "*": 3,
            "/": 3,
            "+": 2,
            "-": 2,
        }
        
        expr_prec = precedence.get(expr.op, 0)
        parent_prec = precedence.get(parent_op, 0)
        
        # Lower precedence always needs parentheses
        if expr_prec < parent_prec:
            return True
        
        # For equal precedence, check associativity
        if expr_prec == parent_prec:
            # Right side of non-associative operators needs parentheses
            if is_right and parent_op in ["-", "/", "^"]:
                return True
        
        return False
    
    return False


def binary_to_string(binary_op: "BinaryOp") -> str:
    """Convert a binary operation to a string."""
    left_str = binary_op.left.to_string()
    right_str = binary_op.right.to_string()
    
    # Add parentheses if needed
    if needs_parentheses(binary_op.left, binary_op.op, False):
        left_str = f"({left_str})"
    
    if needs_parentheses(binary_op.right, binary_op.op, True):
        right_str = f"({right_str})"
    
    return f"{left_str} {binary_op.op} {right_str}"


def binary_to_latex(binary_op: "BinaryOp") -> str:
    """Convert a binary operation to LaTeX format."""
    from symbolic.ast import Constant
    
    left_latex = binary_op.left.to_latex()
    right_latex = binary_op.right.to_latex()
    
    # Handle division as fraction
    if binary_op.op == "/":
        return rf"\frac{{{left_latex}}}{{{right_latex}}}"
    
    # Handle power
    if binary_op.op == "^":
        # Add braces for left side if needed
        if needs_parentheses(binary_op.left, binary_op.op, False):
            left_latex = f"({left_latex})"
        return f"{left_latex}^{{{right_latex}}}"
    
    # Handle multiplication - use \cdot
    if binary_op.op == "*":
        if needs_parentheses(binary_op.left, binary_op.op, False):
            left_latex = f"({left_latex})"
        if needs_parentheses(binary_op.right, binary_op.op, True):
            right_latex = f"({right_latex})"
        
        # Implicit multiplication for certain cases
        from symbolic.ast import Variable, Function
        if isinstance(binary_op.left, Constant) and isinstance(binary_op.right, (Variable, Function)):
            return f"{left_latex}{right_latex}"
        
        return f"{left_latex} \\cdot {right_latex}"
    
    # Handle addition and subtraction
    if needs_parentheses(binary_op.left, binary_op.op, False):
        left_latex = f"({left_latex})"
    
    if needs_parentheses(binary_op.right, binary_op.op, True):
        right_latex = f"({right_latex})"
    
    return f"{left_latex} {binary_op.op} {right_latex}"


def unary_to_string(unary_op: "UnaryOp") -> str:
    """Convert a unary operation to a string."""
    operand_str = unary_op.operand.to_string()
    
    from symbolic.ast import BinaryOp
    # Add parentheses for binary operations
    if isinstance(unary_op.operand, BinaryOp):
        operand_str = f"({operand_str})"
    
    if unary_op.op == "-":
        return f"-{operand_str}"
    elif unary_op.op == "+":
        return f"+{operand_str}"
    
    return f"{unary_op.op}{operand_str}"


def unary_to_latex(unary_op: "UnaryOp") -> str:
    """Convert a unary operation to LaTeX format."""
    operand_latex = unary_op.operand.to_latex()
    
    from symbolic.ast import BinaryOp
    # Add parentheses for binary operations
    if isinstance(unary_op.operand, BinaryOp):
        operand_latex = f"({operand_latex})"
    
    if unary_op.op == "-":
        return f"-{operand_latex}"
    elif unary_op.op == "+":
        return f"+{operand_latex}"
    
    return f"{unary_op.op}{operand_latex}"


def function_to_string(func: "Function") -> str:
    """Convert a function to a string."""
    arg_str = func.arg.to_string()
    return f"{func.name}({arg_str})"


def function_to_latex(func: "Function") -> str:
    """Convert a function to LaTeX format."""
    arg_latex = func.arg.to_latex()
    
    # Standard LaTeX function names
    latex_functions = {"sin", "cos", "tan", "log", "exp", "sqrt"}
    
    if func.name in latex_functions:
        if func.name == "sqrt":
            return rf"\sqrt{{{arg_latex}}}"
        return rf"\{func.name}({arg_latex})"
    
    # Custom functions
    return f"{func.name}({arg_latex})"
