"""
Mathematical expression parser for the Math-Music-Engine.

This module provides functionality to parse mathematical expressions and convert them
into executable functions using SymPy.
"""

from typing import Callable, Dict, Any, Optional, Tuple
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np


class ExpressionParser:
    """
    Parse mathematical expressions and convert them to executable functions.
    
    Supports constants, variables, unary and binary operations.
    Detects domain and continuity constraints.
    """
    
    def __init__(self):
        """Initialize the expression parser."""
        self.transformations = standard_transformations + (implicit_multiplication_application,)
        self.cached_functions: Dict[str, Tuple[sp.Expr, Callable]] = {}
        
    def parse(
        self, 
        expression: str, 
        variables: Optional[list[str]] = None
    ) -> Tuple[sp.Expr, Dict[str, Any]]:
        """
        Parse a mathematical expression string into a SymPy expression.
        
        Args:
            expression: Mathematical expression as a string (e.g., "sin(2*pi*t)")
            variables: List of variable names. If None, automatically detects variables.
            
        Returns:
            Tuple of (SymPy expression, metadata dict)
            
        Raises:
            ValueError: If expression cannot be parsed
        """
        try:
            # Parse the expression
            expr = parse_expr(expression, transformations=self.transformations)
            
            # Detect variables if not provided
            if variables is None:
                detected_vars = sorted([str(s) for s in expr.free_symbols])
            else:
                detected_vars = variables
                
            # Analyze the expression
            metadata = self._analyze_expression(expr, detected_vars)
            metadata['original_expression'] = expression
            
            return expr, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to parse expression '{expression}': {str(e)}")
    
    def _analyze_expression(self, expr: sp.Expr, variables: list[str]) -> Dict[str, Any]:
        """
        Analyze mathematical properties of the expression.
        
        Args:
            expr: SymPy expression
            variables: List of variable names
            
        Returns:
            Dictionary containing metadata about the expression
        """
        metadata = {
            'variables': variables,
            'is_polynomial': expr.is_polynomial(),
            'is_rational': expr.is_rational_function(),
        }
        
        # Check for common function types
        metadata['contains_trig'] = any(func in str(expr) for func in ['sin', 'cos', 'tan'])
        metadata['contains_exp'] = 'exp' in str(expr) or 'E**' in str(expr)
        metadata['contains_log'] = 'log' in str(expr)
        
        # Try to determine domain constraints
        if len(variables) == 1:
            var = sp.Symbol(variables[0])
            try:
                # Check for singularities
                singularities = sp.singularities(expr, var)
                metadata['singularities'] = [float(s.evalf()) if s.is_real else str(s) 
                                            for s in singularities if s.is_finite]
            except:
                metadata['singularities'] = []
        else:
            metadata['singularities'] = []
            
        return metadata
    
    def to_numpy_function(
        self, 
        expression: str, 
        variables: Optional[list[str]] = None,
        use_cache: bool = True
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Convert a mathematical expression to a NumPy-compatible function.
        
        Args:
            expression: Mathematical expression as a string
            variables: List of variable names (default: auto-detect)
            use_cache: Whether to cache the compiled function
            
        Returns:
            Tuple of (callable function, metadata dict)
        """
        # Check cache
        if use_cache and expression in self.cached_functions:
            cached_expr, cached_func = self.cached_functions[expression]
            _, metadata = self.parse(expression, variables)
            return cached_func, metadata
        
        # Parse expression
        expr, metadata = self.parse(expression, variables)
        
        # Create symbols
        var_symbols = [sp.Symbol(v) for v in metadata['variables']]
        
        # Convert to NumPy function using lambdify
        try:
            func = sp.lambdify(
                var_symbols, 
                expr, 
                modules=['numpy', 'sympy']
            )
            
            # Wrap to ensure proper array handling
            def wrapped_func(*args):
                result = func(*args)
                return np.asarray(result, dtype=np.float64)
            
            # Cache the function
            if use_cache:
                self.cached_functions[expression] = (expr, wrapped_func)
            
            return wrapped_func, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to create callable function: {str(e)}")
    
    def differentiate(
        self, 
        expression: str, 
        variable: str,
        order: int = 1
    ) -> str:
        """
        Compute the derivative of an expression.
        
        Args:
            expression: Mathematical expression as a string
            variable: Variable to differentiate with respect to
            order: Order of differentiation (default: 1)
            
        Returns:
            Derivative expression as a string
        """
        expr, _ = self.parse(expression)
        var = sp.Symbol(variable)
        
        derivative = sp.diff(expr, var, order)
        return str(derivative)
    
    def integrate(
        self, 
        expression: str, 
        variable: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> str:
        """
        Compute the integral of an expression.
        
        Args:
            expression: Mathematical expression as a string
            variable: Variable to integrate with respect to
            lower: Lower bound for definite integral (optional)
            upper: Upper bound for definite integral (optional)
            
        Returns:
            Integral expression as a string (or numerical value for definite integral)
        """
        expr, _ = self.parse(expression)
        var = sp.Symbol(variable)
        
        if lower is not None and upper is not None:
            # Definite integral
            integral = sp.integrate(expr, (var, lower, upper))
            try:
                return str(float(integral.evalf()))
            except:
                return str(integral)
        else:
            # Indefinite integral
            integral = sp.integrate(expr, var)
            return str(integral)
    
    def simplify(self, expression: str) -> str:
        """
        Simplify a mathematical expression.
        
        Args:
            expression: Mathematical expression as a string
            
        Returns:
            Simplified expression as a string
        """
        expr, _ = self.parse(expression)
        simplified = sp.simplify(expr)
        return str(simplified)
    
    def substitute(
        self, 
        expression: str, 
        substitutions: Dict[str, float]
    ) -> float:
        """
        Substitute values into an expression and evaluate.
        
        Args:
            expression: Mathematical expression as a string
            substitutions: Dictionary mapping variable names to values
            
        Returns:
            Numerical result
        """
        expr, _ = self.parse(expression)
        
        # Create substitution dict with SymPy symbols
        subs_dict = {sp.Symbol(k): v for k, v in substitutions.items()}
        
        result = expr.subs(subs_dict)
        return float(result.evalf())


# Convenience functions
def parse_expression(expression: str) -> Tuple[sp.Expr, Dict[str, Any]]:
    """
    Convenience function to parse an expression.
    
    Args:
        expression: Mathematical expression as a string
        
    Returns:
        Tuple of (SymPy expression, metadata dict)
    """
    parser = ExpressionParser()
    return parser.parse(expression)


def expression_to_function(expression: str, variables: Optional[list[str]] = None) -> Callable:
    """
    Convenience function to convert expression to callable function.
    
    Args:
        expression: Mathematical expression as a string
        variables: List of variable names (default: auto-detect)
        
    Returns:
        Callable NumPy-compatible function
    """
    parser = ExpressionParser()
    func, _ = parser.to_numpy_function(expression, variables)
    return func
