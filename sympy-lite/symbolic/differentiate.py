"""
Symbolic differentiation engine.

Implements recursive differentiation rules including:
- Basic rules (constants, variables)
- Sum and difference rules
- Product rule
- Quotient rule
- Chain rule
- Power rule
- Trigonometric and exponential functions
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from symbolic.ast import Expression, BinaryOp, UnaryOp, Function

from symbolic.ast import Constant, BinaryOp, UnaryOp, Function


def differentiate_binary(binary_op: "BinaryOp", var: str) -> "Expression":
    """Apply differentiation rules to binary operations."""
    op = binary_op.op
    u = binary_op.left
    v = binary_op.right
    
    # Get derivatives of operands
    du = u.differentiate(var)
    dv = v.differentiate(var)
    
    # Sum rule: d(u + v)/dx = du/dx + dv/dx
    if op == "+":
        result = BinaryOp("+", du, dv)
        return result.simplify()
    
    # Difference rule: d(u - v)/dx = du/dx - dv/dx
    if op == "-":
        result = BinaryOp("-", du, dv)
        return result.simplify()
    
    # Product rule: d(u * v)/dx = u' * v + u * v'
    if op == "*":
        term1 = BinaryOp("*", du, v)
        term2 = BinaryOp("*", u, dv)
        result = BinaryOp("+", term1, term2)
        return result.simplify()
    
    # Quotient rule: d(u / v)/dx = (u' * v - u * v') / v^2
    if op == "/":
        numerator_term1 = BinaryOp("*", du, v)
        numerator_term2 = BinaryOp("*", u, dv)
        numerator = BinaryOp("-", numerator_term1, numerator_term2)
        denominator = BinaryOp("^", v, Constant(2.0))
        result = BinaryOp("/", numerator, denominator)
        return result.simplify()
    
    # Power rule: d(u^v)/dx
    if op == "^":
        # If v is constant: d(u^n)/dx = n * u^(n-1) * u'
        if not v.contains_var(var):
            # n * u^(n-1) * u'
            exponent = BinaryOp("-", v, Constant(1.0))
            power = BinaryOp("^", u, exponent)
            term = BinaryOp("*", v, power)
            result = BinaryOp("*", term, du)
            return result.simplify()
        
        # If u is constant: d(a^v)/dx = a^v * ln(a) * v'
        if not u.contains_var(var):
            # a^v * ln(a) * v'
            power = BinaryOp("^", u, v)
            log_u = Function("log", u)
            term = BinaryOp("*", power, log_u)
            result = BinaryOp("*", term, dv)
            return result.simplify()
        
        # General case: d(u^v)/dx = u^v * (v' * ln(u) + v * u'/u)
        # This is the full exponential differentiation formula
        power = BinaryOp("^", u, v)
        log_u = Function("log", u)
        term1 = BinaryOp("*", dv, log_u)
        u_over_u = BinaryOp("/", du, u)
        term2 = BinaryOp("*", v, u_over_u)
        sum_terms = BinaryOp("+", term1, term2)
        result = BinaryOp("*", power, sum_terms)
        return result.simplify()
    
    # If we reach here, something went wrong
    from symbolic.errors import SimplificationError
    raise SimplificationError(f"Unknown binary operator for differentiation: {op}")


def differentiate_unary(unary_op: "UnaryOp", var: str) -> "Expression":
    """Apply differentiation rules to unary operations."""
    operand = unary_op.operand
    d_operand = operand.differentiate(var)
    
    # d(-u)/dx = -du/dx
    if unary_op.op == "-":
        result = UnaryOp("-", d_operand)
        return result.simplify()
    
    # d(+u)/dx = du/dx
    if unary_op.op == "+":
        return d_operand.simplify()
    
    from symbolic.errors import SimplificationError
    raise SimplificationError(f"Unknown unary operator for differentiation: {unary_op.op}")


def differentiate_function(func: "Function", var: str) -> "Expression":
    """Apply differentiation rules to functions using the chain rule."""
    u = func.arg
    du = u.differentiate(var)
    
    # Chain rule: d(f(u))/dx = f'(u) * u'
    
    # d(sin(u))/dx = cos(u) * u'
    if func.name == "sin":
        cos_u = Function("cos", u)
        result = BinaryOp("*", cos_u, du)
        return result.simplify()
    
    # d(cos(u))/dx = -sin(u) * u'
    if func.name == "cos":
        sin_u = Function("sin", u)
        neg_sin = UnaryOp("-", sin_u)
        result = BinaryOp("*", neg_sin, du)
        return result.simplify()
    
    # d(tan(u))/dx = sec^2(u) * u' = (1/cos^2(u)) * u'
    if func.name == "tan":
        cos_u = Function("cos", u)
        cos_squared = BinaryOp("^", cos_u, Constant(2.0))
        sec_squared = BinaryOp("/", Constant(1.0), cos_squared)
        result = BinaryOp("*", sec_squared, du)
        return result.simplify()
    
    # d(exp(u))/dx = exp(u) * u'
    if func.name == "exp":
        exp_u = Function("exp", u)
        result = BinaryOp("*", exp_u, du)
        return result.simplify()
    
    # d(log(u))/dx = (1/u) * u'
    if func.name == "log":
        one_over_u = BinaryOp("/", Constant(1.0), u)
        result = BinaryOp("*", one_over_u, du)
        return result.simplify()
    
    # d(sqrt(u))/dx = (1/(2*sqrt(u))) * u'
    if func.name == "sqrt":
        sqrt_u = Function("sqrt", u)
        two_sqrt_u = BinaryOp("*", Constant(2.0), sqrt_u)
        derivative = BinaryOp("/", Constant(1.0), two_sqrt_u)
        result = BinaryOp("*", derivative, du)
        return result.simplify()
    
    # d(abs(u))/dx = (u/abs(u)) * u' for u != 0
    if func.name == "abs":
        abs_u = Function("abs", u)
        sign = BinaryOp("/", u, abs_u)
        result = BinaryOp("*", sign, du)
        return result.simplify()
    
    from symbolic.errors import SimplificationError
    raise SimplificationError(f"Unknown function for differentiation: {func.name}")
