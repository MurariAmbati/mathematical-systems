"""
Symbolic integration engine.

Implements basic integration rules for:
- Polynomials
- Exponential functions
- Trigonometric functions
- Linear combinations
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbolic.ast import Expression, BinaryOp, UnaryOp, Function

from symbolic.ast import Constant, Variable, BinaryOp, UnaryOp, Function
from symbolic.errors import IntegrationError


def integrate_binary(binary_op: "BinaryOp", var: str) -> "Expression":
    """Apply integration rules to binary operations."""
    op = binary_op.op
    u = binary_op.left
    v = binary_op.right
    
    # Linearity: ∫(u + v) dx = ∫u dx + ∫v dx
    if op == "+":
        int_u = u.integrate(var)
        int_v = v.integrate(var)
        result = BinaryOp("+", int_u, int_v)
        return result.simplify()
    
    # Linearity: ∫(u - v) dx = ∫u dx - ∫v dx
    if op == "-":
        int_u = u.integrate(var)
        int_v = v.integrate(var)
        result = BinaryOp("-", int_u, int_v)
        return result.simplify()
    
    # Constant multiple: ∫(c * u) dx = c * ∫u dx
    if op == "*":
        if not u.contains_var(var):
            # u is constant with respect to var
            int_v = v.integrate(var)
            result = BinaryOp("*", u, int_v)
            return result.simplify()
        elif not v.contains_var(var):
            # v is constant with respect to var
            int_u = u.integrate(var)
            result = BinaryOp("*", int_u, v)
            return result.simplify()
        else:
            # Both contain var - would need integration by parts
            raise IntegrationError(
                f"Cannot integrate product of two variable expressions: {binary_op.to_string()}"
            )
    
    # Division by constant: ∫(u / c) dx = (1/c) * ∫u dx
    if op == "/":
        if not v.contains_var(var):
            int_u = u.integrate(var)
            result = BinaryOp("/", int_u, v)
            return result.simplify()
        else:
            raise IntegrationError(
                f"Cannot integrate division with variable in denominator: {binary_op.to_string()}"
            )
    
    # Power rule for x^n: ∫x^n dx = x^(n+1)/(n+1)
    if op == "^":
        # Check if this is x^n where x is the variable and n is constant
        if isinstance(u, Variable) and u.name == var and not v.contains_var(var):
            # ∫x^n dx = x^(n+1)/(n+1), n ≠ -1
            # Check if n = -1 (special case: ∫x^(-1) dx = ln(x))
            if isinstance(v, Constant) and v.value == -1:
                result = Function("log", u)
                return result.simplify()
            
            new_exponent = BinaryOp("+", v, Constant(1.0))
            power = BinaryOp("^", u, new_exponent)
            result = BinaryOp("/", power, new_exponent)
            return result.simplify()
        else:
            raise IntegrationError(
                f"Cannot integrate general power expression: {binary_op.to_string()}"
            )
    
    raise IntegrationError(f"Cannot integrate binary operation: {binary_op.to_string()}")


def integrate_unary(unary_op: "UnaryOp", var: str) -> "Expression":
    """Apply integration rules to unary operations."""
    operand = unary_op.operand
    
    # ∫(-u) dx = -∫u dx
    if unary_op.op == "-":
        int_operand = operand.integrate(var)
        result = UnaryOp("-", int_operand)
        return result.simplify()
    
    # ∫(+u) dx = ∫u dx
    if unary_op.op == "+":
        result = operand.integrate(var)
        return result.simplify()
    
    raise IntegrationError(f"Unknown unary operator for integration: {unary_op.op}")


def integrate_function(func: "Function", var: str) -> "Expression":
    """Apply integration rules to functions."""
    u = func.arg
    
    # Simple cases where argument is just the variable
    if isinstance(u, Variable) and u.name == var:
        # ∫sin(x) dx = -cos(x)
        if func.name == "sin":
            cos_x = Function("cos", u)
            result = UnaryOp("-", cos_x)
            return result.simplify()
        
        # ∫cos(x) dx = sin(x)
        if func.name == "cos":
            result = Function("sin", u)
            return result.simplify()
        
        # ∫exp(x) dx = exp(x)
        if func.name == "exp":
            result = Function("exp", u)
            return result.simplify()
        
        # ∫(1/x) dx = log(x) - handled via x^(-1) in power rule
        # But we can also handle log directly if needed
        
        # ∫tan(x) dx = -log(cos(x))
        if func.name == "tan":
            cos_x = Function("cos", u)
            log_cos = Function("log", cos_x)
            result = UnaryOp("-", log_cos)
            return result.simplify()
    
    # More complex cases: u-substitution needed
    # For now, handle linear substitution: ∫f(ax+b) dx
    
    # Check if argument is linear: ax + b or ax or x + b
    if isinstance(u, BinaryOp):
        # Try to detect ax + b or ax form
        if u.op == "*" and isinstance(u.left, Constant) and isinstance(u.right, Variable):
            if u.right.name == var:
                # Form: a*x
                a = u.left
                
                if func.name == "sin":
                    # ∫sin(ax) dx = -(1/a)*cos(ax)
                    cos_ax = Function("cos", u)
                    one_over_a = BinaryOp("/", Constant(1.0), a)
                    result = UnaryOp("-", BinaryOp("*", one_over_a, cos_ax))
                    return result.simplify()
                
                if func.name == "cos":
                    # ∫cos(ax) dx = (1/a)*sin(ax)
                    sin_ax = Function("sin", u)
                    one_over_a = BinaryOp("/", Constant(1.0), a)
                    result = BinaryOp("*", one_over_a, sin_ax)
                    return result.simplify()
                
                if func.name == "exp":
                    # ∫exp(ax) dx = (1/a)*exp(ax)
                    exp_ax = Function("exp", u)
                    one_over_a = BinaryOp("/", Constant(1.0), a)
                    result = BinaryOp("*", one_over_a, exp_ax)
                    return result.simplify()
    
    raise IntegrationError(
        f"Cannot integrate function {func.name} with argument {u.to_string()}"
    )
