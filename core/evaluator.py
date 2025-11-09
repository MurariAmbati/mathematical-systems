"""
Expression Evaluator

Handles symbolic and numeric evaluation with optimization.
"""

from typing import Dict, Union, Optional, Callable, Any
import numpy as np
from numpy.typing import NDArray
import sympy as sp
from core.parser import ParsedExpression, parse_equation


class Evaluator:
    """
    High-performance evaluator for mathematical expressions.
    
    Supports both symbolic and numeric evaluation with caching.
    """
    
    def __init__(
        self,
        expression: Union[str, ParsedExpression],
        backend: str = "numpy"
    ):
        """
        Initialize evaluator.
        
        Args:
            expression: Mathematical expression (string or ParsedExpression)
            backend: "numpy" for vectorized ops, "numba" for JIT compilation
        """
        if isinstance(expression, str):
            self.parsed_expr = parse_equation(expression)
        else:
            self.parsed_expr = expression
        
        self.backend = backend
        self._compiled_func = None
    
    def compile(self) -> Callable:
        """Compile the expression to a fast callable."""
        if self._compiled_func is None:
            self._compiled_func = self.parsed_expr.compile(backend=self.backend)
        return self._compiled_func
    
    def evaluate(
        self,
        **variables: Union[float, NDArray]
    ) -> Union[float, NDArray]:
        """
        Evaluate the expression with given variable values.
        
        Args:
            **variables: Variable names and values (scalars or arrays)
            
        Returns:
            Evaluation result (scalar or array)
            
        Example:
            >>> evaluator = Evaluator("sin(x) + cos(y)")
            >>> result = evaluator.evaluate(x=1.0, y=2.0)
            >>> # Or with arrays
            >>> x = np.linspace(0, np.pi, 100)
            >>> result = evaluator.evaluate(x=x, y=0)
        """
        if self._compiled_func is None:
            self.compile()
        
        return self._compiled_func(**variables)
    
    def evaluate_grid(
        self,
        x_range: tuple,
        y_range: tuple,
        samples: int = 100
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Evaluate expression over a 2D grid.
        
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            samples: Number of samples in each dimension
            
        Returns:
            (X, Y, Z) meshgrid arrays where Z = f(X, Y)
        """
        x = np.linspace(x_range[0], x_range[1], samples)
        y = np.linspace(y_range[0], y_range[1], samples)
        X, Y = np.meshgrid(x, y)
        
        Z = self.evaluate(x=X, y=Y)
        
        return X, Y, Z
    
    def differentiate(
        self,
        variable: str,
        order: int = 1
    ) -> 'Evaluator':
        """
        Compute symbolic derivative.
        
        Args:
            variable: Variable to differentiate with respect to
            order: Order of derivative (1 for first, 2 for second, etc.)
            
        Returns:
            New Evaluator with derivative expression
        """
        var_symbol = sp.Symbol(variable)
        
        derivative_expr = self.parsed_expr.sympy_expr
        for _ in range(order):
            derivative_expr = sp.diff(derivative_expr, var_symbol)
        
        # Create new ParsedExpression
        from core.parser import ParsedExpression
        derivative_parsed = ParsedExpression(
            expr_str=str(derivative_expr),
            sympy_expr=derivative_expr,
            variables=self.parsed_expr.variables,
            is_safe=True
        )
        
        return Evaluator(derivative_parsed, backend=self.backend)
    
    def integrate(
        self,
        variable: str,
        limits: Optional[tuple] = None
    ) -> Union['Evaluator', float]:
        """
        Compute symbolic or numeric integration.
        
        Args:
            variable: Variable to integrate with respect to
            limits: (lower, upper) bounds for definite integral
                   If None, returns indefinite integral
            
        Returns:
            Evaluator with integrated expression, or numeric result if definite
        """
        var_symbol = sp.Symbol(variable)
        
        if limits is None:
            # Indefinite integral
            integral_expr = sp.integrate(self.parsed_expr.sympy_expr, var_symbol)
            
            from core.parser import ParsedExpression
            integral_parsed = ParsedExpression(
                expr_str=str(integral_expr),
                sympy_expr=integral_expr,
                variables=self.parsed_expr.variables,
                is_safe=True
            )
            
            return Evaluator(integral_parsed, backend=self.backend)
        else:
            # Definite integral
            lower, upper = limits
            result = sp.integrate(
                self.parsed_expr.sympy_expr,
                (var_symbol, lower, upper)
            )
            return float(result)
    
    def simplify(self) -> 'Evaluator':
        """
        Simplify the expression symbolically.
        
        Returns:
            New Evaluator with simplified expression
        """
        simplified_expr = sp.simplify(self.parsed_expr.sympy_expr)
        
        from core.parser import ParsedExpression
        simplified_parsed = ParsedExpression(
            expr_str=str(simplified_expr),
            sympy_expr=simplified_expr,
            variables=self.parsed_expr.variables,
            is_safe=True
        )
        
        return Evaluator(simplified_parsed, backend=self.backend)
    
    def __repr__(self) -> str:
        return f"Evaluator('{self.parsed_expr.expr_str}')"


def evaluate_batch(
    expressions: list[Union[str, ParsedExpression]],
    **variables: Union[float, NDArray]
) -> list[Union[float, NDArray]]:
    """
    Evaluate multiple expressions with the same variable values.
    
    Args:
        expressions: List of expressions to evaluate
        **variables: Variable names and values
        
    Returns:
        List of evaluation results
        
    Example:
        >>> exprs = ["sin(x)", "cos(x)", "tan(x)"]
        >>> results = evaluate_batch(exprs, x=np.linspace(0, np.pi, 100))
    """
    evaluators = [Evaluator(expr) for expr in expressions]
    results = [evaluator.evaluate(**variables) for evaluator in evaluators]
    return results


def create_parametric_evaluator(
    fx_expr: str,
    fy_expr: str,
    fz_expr: Optional[str] = None
) -> Callable:
    """
    Create a parametric evaluator for curves/surfaces.
    
    Args:
        fx_expr: Expression for x component
        fy_expr: Expression for y component
        fz_expr: Expression for z component (optional, for 3D)
        
    Returns:
        Function that takes parameter value(s) and returns points
        
    Example:
        >>> # Lissajous curve
        >>> evaluator = create_parametric_evaluator("sin(3*t)", "sin(2*t)")
        >>> t = np.linspace(0, 2*np.pi, 1000)
        >>> points = evaluator(t=t)  # Returns (1000, 2) array
    """
    fx_eval = Evaluator(fx_expr)
    fy_eval = Evaluator(fy_expr)
    fz_eval = Evaluator(fz_expr) if fz_expr else None
    
    def parametric_func(**params) -> NDArray:
        x = fx_eval.evaluate(**params)
        y = fy_eval.evaluate(**params)
        
        if fz_eval is not None:
            z = fz_eval.evaluate(**params)
            return np.column_stack([x, y, z])
        else:
            return np.column_stack([x, y])
    
    return parametric_func
