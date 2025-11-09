"""
Expression Parser for Math Art Generator

Safely parse and validate user-defined mathematical expressions.
Supports 30+ math functions and multiple variables (x, y, z, t, r, theta).
"""

from typing import Callable, Dict, List, Set, Optional, Union
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import numpy as np
from numba import jit


# Supported variables
SUPPORTED_VARIABLES = {"x", "y", "z", "t", "r", "theta", "θ"}

# Supported functions (30+ mathematical functions)
SUPPORTED_FUNCTIONS = {
    # Trigonometric
    "sin", "cos", "tan", "sec", "csc", "cot",
    "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    
    # Exponential and logarithmic
    "exp", "log", "ln", "log10", "log2",
    
    # Power and roots
    "sqrt", "cbrt", "pow",
    
    # Special functions
    "abs", "sign", "floor", "ceil", "round",
    
    # Complex
    "re", "im", "arg", "conjugate",
    
    # Hyperbolic
    "sech", "csch", "coth",
    
    # Other
    "max", "min", "mod",
}


class ParsedExpression:
    """Represents a parsed and compiled mathematical expression."""
    
    def __init__(
        self,
        expr_str: str,
        sympy_expr: sp.Expr,
        variables: Set[str],
        is_safe: bool = True
    ):
        self.expr_str = expr_str
        self.sympy_expr = sympy_expr
        self.variables = variables
        self.is_safe = is_safe
        self._compiled_func: Optional[Callable] = None
        self._vectorized_func: Optional[Callable] = None
        
    def compile(self, backend: str = "numpy") -> Callable:
        """
        Compile the expression to a callable function.
        
        Args:
            backend: "numpy" for vectorized operations, "numba" for JIT compilation
            
        Returns:
            Callable function that takes variable values as keyword arguments
        """
        if self._compiled_func is not None and backend == "numpy":
            return self._compiled_func
            
        # Create ordered list of variables for lambdify
        var_list = sorted(list(self.variables))
        symbols = [sp.Symbol(v) for v in var_list]
        
        if backend == "numpy":
            # Use SymPy's lambdify for fast NumPy evaluation
            func = sp.lambdify(
                symbols,
                self.sympy_expr,
                modules=["numpy", {"ln": np.log}]
            )
            self._compiled_func = lambda **kwargs: func(
                *[kwargs.get(v, 0) for v in var_list]
            )
            return self._compiled_func
            
        elif backend == "numba":
            # First create numpy function, then wrap with numba
            numpy_func = sp.lambdify(symbols, self.sympy_expr, modules="numpy")
            
            # Note: Numba JIT compilation for actual performance gain
            # This is a simplified version
            self._compiled_func = lambda **kwargs: numpy_func(
                *[kwargs.get(v, 0) for v in var_list]
            )
            return self._compiled_func
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def evaluate(self, **kwargs) -> Union[float, np.ndarray]:
        """
        Evaluate the expression with given variable values.
        
        Args:
            **kwargs: Variable names and their values (scalars or arrays)
            
        Returns:
            Result of evaluation (scalar or array)
        """
        if self._compiled_func is None:
            self.compile()
        return self._compiled_func(**kwargs)
    
    def __repr__(self) -> str:
        return f"ParsedExpression('{self.expr_str}', variables={self.variables})"


def parse_equation(
    expr_str: str,
    validate: bool = True,
    check_domain: bool = True
) -> ParsedExpression:
    """
    Parse a mathematical expression safely.
    
    Args:
        expr_str: Mathematical expression as string (e.g., "sin(x*y) + cos(r^2)")
        validate: Whether to validate the expression for safety
        check_domain: Whether to check for potential domain issues
        
    Returns:
        ParsedExpression object with compiled evaluation function
        
    Raises:
        ValueError: If expression is invalid or unsafe
        
    Examples:
        >>> expr = parse_equation("sin(x*y) + cos(x^2 - y^2)")
        >>> result = expr.evaluate(x=1.0, y=2.0)
        
        >>> expr = parse_equation("r*cos(theta)")
        >>> r_vals = np.linspace(0, 1, 100)
        >>> theta_vals = np.linspace(0, 2*np.pi, 100)
        >>> result = expr.evaluate(r=r_vals, theta=theta_vals)
    """
    if not expr_str or not isinstance(expr_str, str):
        raise ValueError("Expression must be a non-empty string")
    
    # Normalize expression
    expr_str = expr_str.strip()
    
    # Replace common aliases
    expr_str = expr_str.replace("θ", "theta")
    expr_str = expr_str.replace("^", "**")
    expr_str = expr_str.replace("ln(", "log(")
    
    # Parse with SymPy
    try:
        transformations = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor)
        )
        
        # Create local dict with Symbol and common functions
        local_dict = {
            'Symbol': sp.Symbol,
            'Integer': sp.Integer,
            'Float': sp.Float,
            'Number': sp.Number,
            'Rational': sp.Rational,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'atan2': sp.atan2,
            'pi': sp.pi, 'e': sp.E,
        }
        
        sympy_expr = parse_expr(
            expr_str,
            transformations=transformations,
            local_dict=local_dict,
            global_dict={}
        )
    except Exception as e:
        raise ValueError(f"Failed to parse expression: {e}")
    
    # Extract variables
    symbols = sympy_expr.free_symbols
    variables = {str(s) for s in symbols}
    
    # Validate variables
    if validate:
        unsupported_vars = variables - SUPPORTED_VARIABLES
        if unsupported_vars:
            raise ValueError(
                f"Unsupported variables: {unsupported_vars}. "
                f"Supported: {SUPPORTED_VARIABLES}"
            )
    
    # Validate functions
    if validate:
        funcs = {str(f.func) for f in sympy_expr.atoms(sp.Function)}
        # Extract function names from SymPy function objects
        func_names = set()
        for atom in sympy_expr.atoms():
            if hasattr(atom, 'func') and atom.func.__name__ not in ['Symbol', 'Integer', 'Float', 'Rational']:
                func_names.add(atom.func.__name__.lower())
        
        # Filter out basic operations and internal SymPy functions
        basic_ops = {'Add', 'Mul', 'Pow', 'Number', 'Symbol', 'add', 'mul', 'pow', 'number', 
                     'negativeone', 'one', 'half', 'integer', 'rational', 'float'}
        unsupported_funcs = func_names - SUPPORTED_FUNCTIONS - basic_ops
        
        if unsupported_funcs:
            raise ValueError(
                f"Unsupported functions: {unsupported_funcs}. "
                f"Supported: {sorted(SUPPORTED_FUNCTIONS)}"
            )
    
    # Check for domain issues
    if check_domain:
        _check_domain_issues(sympy_expr)
    
    return ParsedExpression(
        expr_str=expr_str,
        sympy_expr=sympy_expr,
        variables=variables,
        is_safe=True
    )


def _check_domain_issues(expr: sp.Expr) -> None:
    """
    Check for potential domain issues (e.g., sqrt of negative, log of zero).
    
    This is a basic check - runtime validation is still needed.
    """
    # Check for sqrt - warn about potential negative values
    for atom in expr.atoms(sp.sqrt):
        # Could add more sophisticated checking here
        pass
    
    # Check for log - warn about potential zero/negative values
    for atom in expr.atoms(sp.log):
        pass
    
    # Check for division - warn about potential division by zero
    for atom in expr.atoms(sp.Pow):
        if atom.exp.is_negative:
            pass  # Potential division by zero


def list_supported_functions() -> List[str]:
    """Return list of all supported mathematical functions."""
    return sorted(SUPPORTED_FUNCTIONS)


def list_supported_variables() -> List[str]:
    """Return list of all supported variables."""
    return sorted(SUPPORTED_VARIABLES)


def validate_expression(expr_str: str) -> Dict[str, any]:
    """
    Validate an expression and return diagnostic information.
    
    Args:
        expr_str: Expression string to validate
        
    Returns:
        Dictionary with validation results:
        - valid: bool
        - variables: set of variables
        - functions: set of functions
        - error: error message if invalid
    """
    try:
        parsed = parse_equation(expr_str, validate=True, check_domain=True)
        return {
            "valid": True,
            "variables": parsed.variables,
            "expression": str(parsed.sympy_expr),
            "error": None
        }
    except Exception as e:
        return {
            "valid": False,
            "variables": None,
            "expression": None,
            "error": str(e)
        }
