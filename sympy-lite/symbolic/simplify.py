"""
Algebraic simplification rules and engine.

Implements pattern-based simplification including:
- Arithmetic identities (x+0=x, x*1=x, x*0=0)
- Constant folding
- Power rules
- Nested unary simplification
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbolic.ast import Expression, BinaryOp, UnaryOp, Function

from symbolic.ast import Constant, Variable, BinaryOp, UnaryOp


def simplify_binary(binary_op: "BinaryOp") -> "Expression":
    """Apply simplification rules to binary operations."""
    # First, recursively simplify children
    left = binary_op.left.simplify()
    right = binary_op.right.simplify()
    
    op = binary_op.op
    
    # Constant folding
    if isinstance(left, Constant) and isinstance(right, Constant):
        if op == "+":
            return Constant(left.value + right.value)
        elif op == "-":
            return Constant(left.value - right.value)
        elif op == "*":
            return Constant(left.value * right.value)
        elif op == "/":
            if right.value != 0:
                return Constant(left.value / right.value)
        elif op == "^":
            return Constant(left.value ** right.value)
    
    # Addition rules
    if op == "+":
        # x + 0 = x
        if isinstance(right, Constant) and right.value == 0:
            return left
        # 0 + x = x
        if isinstance(left, Constant) and left.value == 0:
            return right
    
    # Subtraction rules
    if op == "-":
        # x - 0 = x
        if isinstance(right, Constant) and right.value == 0:
            return left
        # 0 - x = -x
        if isinstance(left, Constant) and left.value == 0:
            return UnaryOp("-", right).simplify()
        # x - x = 0 (if both are the same variable)
        if isinstance(left, Variable) and isinstance(right, Variable):
            if left.name == right.name:
                return Constant(0.0)
    
    # Multiplication rules
    if op == "*":
        # x * 0 = 0
        if isinstance(right, Constant) and right.value == 0:
            return Constant(0.0)
        if isinstance(left, Constant) and left.value == 0:
            return Constant(0.0)
        # x * 1 = x
        if isinstance(right, Constant) and right.value == 1:
            return left
        # 1 * x = x
        if isinstance(left, Constant) and left.value == 1:
            return right
        # x * -1 = -x
        if isinstance(right, Constant) and right.value == -1:
            return UnaryOp("-", left).simplify()
        if isinstance(left, Constant) and left.value == -1:
            return UnaryOp("-", right).simplify()
    
    # Division rules
    if op == "/":
        # 0 / x = 0 (x != 0)
        if isinstance(left, Constant) and left.value == 0:
            return Constant(0.0)
        # x / 1 = x
        if isinstance(right, Constant) and right.value == 1:
            return left
        # x / -1 = -x
        if isinstance(right, Constant) and right.value == -1:
            return UnaryOp("-", left).simplify()
        # x / x = 1 (if both are the same variable)
        if isinstance(left, Variable) and isinstance(right, Variable):
            if left.name == right.name:
                return Constant(1.0)
    
    # Power rules
    if op == "^":
        # x^0 = 1
        if isinstance(right, Constant) and right.value == 0:
            return Constant(1.0)
        # x^1 = x
        if isinstance(right, Constant) and right.value == 1:
            return left
        # 0^x = 0 (x > 0)
        if isinstance(left, Constant) and left.value == 0:
            return Constant(0.0)
        # 1^x = 1
        if isinstance(left, Constant) and left.value == 1:
            return Constant(1.0)
        # (a^b)^c = a^(b*c)
        if isinstance(left, BinaryOp) and left.op == "^":
            inner_base = left.left
            inner_exp = left.right
            new_exp = BinaryOp("*", inner_exp, right).simplify()
            return BinaryOp("^", inner_base, new_exp).simplify()
    
    # If no simplification applied, return new BinaryOp with simplified children
    if left is binary_op.left and right is binary_op.right:
        return binary_op
    return BinaryOp(op, left, right)


def simplify_unary(unary_op: "UnaryOp") -> "Expression":
    """Apply simplification rules to unary operations."""
    # First, recursively simplify the operand
    operand = unary_op.operand.simplify()
    
    # Constant folding
    if isinstance(operand, Constant):
        if unary_op.op == "-":
            return Constant(-operand.value)
        elif unary_op.op == "+":
            return operand
    
    # Double negation: --x = x
    if unary_op.op == "-" and isinstance(operand, UnaryOp) and operand.op == "-":
        return operand.operand.simplify()
    
    # Unary plus: +x = x
    if unary_op.op == "+":
        return operand
    
    # Negation of zero
    if unary_op.op == "-" and isinstance(operand, Constant) and operand.value == 0:
        return Constant(0.0)
    
    # If no simplification applied, return new UnaryOp with simplified operand
    if operand is unary_op.operand:
        return unary_op
    return UnaryOp(unary_op.op, operand)


def simplify_function(func: "Function") -> "Expression":
    """Apply simplification rules to functions."""
    # First, recursively simplify the argument
    arg = func.arg.simplify()
    
    # Evaluate function with constant argument
    if isinstance(arg, Constant):
        import math
        try:
            if func.name == "sin":
                return Constant(math.sin(arg.value))
            elif func.name == "cos":
                return Constant(math.cos(arg.value))
            elif func.name == "tan":
                return Constant(math.tan(arg.value))
            elif func.name == "exp":
                return Constant(math.exp(arg.value))
            elif func.name == "log":
                if arg.value > 0:
                    return Constant(math.log(arg.value))
            elif func.name == "sqrt":
                if arg.value >= 0:
                    return Constant(math.sqrt(arg.value))
            elif func.name == "abs":
                return Constant(abs(arg.value))
        except (ValueError, OverflowError):
            pass  # Keep as symbolic if evaluation fails
    
    # Specific function simplifications
    # sin(0) = 0, cos(0) = 1, exp(0) = 1, etc.
    if isinstance(arg, Constant):
        if arg.value == 0:
            if func.name == "sin":
                return Constant(0.0)
            elif func.name == "cos":
                return Constant(1.0)
            elif func.name == "exp":
                return Constant(1.0)
            elif func.name == "abs":
                return Constant(0.0)
    
    # If no simplification applied, return new Function with simplified argument
    from symbolic.ast import Function
    if arg is func.arg:
        return func
    return Function(func.name, arg)
