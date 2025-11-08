"""
Abstract Syntax Tree (AST) node definitions for symbolic expressions.

All mathematical expressions are represented as trees of Expression nodes.
Each node type supports simplification, differentiation, integration, and evaluation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import math


class Expression(ABC):
    """Base class for all expression nodes in the AST."""
    
    @abstractmethod
    def simplify(self) -> Expression:
        """Return a simplified version of this expression."""
        pass
    
    @abstractmethod
    def differentiate(self, var: str) -> Expression:
        """Compute the derivative with respect to the given variable."""
        pass
    
    @abstractmethod
    def integrate(self, var: str) -> Expression:
        """Compute the integral with respect to the given variable."""
        pass
    
    @abstractmethod
    def evaluate(self, **values: float) -> float:
        """Evaluate the expression numerically with given variable values."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert the expression to a readable string representation."""
        pass
    
    @abstractmethod
    def to_latex(self) -> str:
        """Convert the expression to LaTeX format."""
        pass
    
    @abstractmethod
    def contains_var(self, var: str) -> bool:
        """Check if this expression contains the given variable."""
        pass
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_string()})"


@dataclass(frozen=True)
class Constant(Expression):
    """Represents a numeric constant (e.g., 3, π, e)."""
    
    value: float
    
    def simplify(self) -> Expression:
        return self
    
    def differentiate(self, var: str) -> Expression:
        return Constant(0.0)
    
    def integrate(self, var: str) -> Expression:
        # ∫c dx = c*x
        return BinaryOp("*", self, Variable(var))
    
    def evaluate(self, **values: float) -> float:
        return self.value
    
    def to_string(self) -> str:
        # Handle special constants
        if self.value == math.pi:
            return "π"
        elif self.value == math.e:
            return "e"
        # Format number nicely
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)
    
    def to_latex(self) -> str:
        if self.value == math.pi:
            return r"\pi"
        elif self.value == math.e:
            return "e"
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)
    
    def contains_var(self, var: str) -> bool:
        return False


@dataclass(frozen=True)
class Variable(Expression):
    """Represents a symbolic variable (e.g., x, y, t)."""
    
    name: str
    
    def simplify(self) -> Expression:
        return self
    
    def differentiate(self, var: str) -> Expression:
        # d(x)/dx = 1, d(y)/dx = 0
        if self.name == var:
            return Constant(1.0)
        return Constant(0.0)
    
    def integrate(self, var: str) -> Expression:
        # ∫x dx = x^2/2, ∫y dx = y*x
        if self.name == var:
            return BinaryOp("/", 
                          BinaryOp("^", self, Constant(2.0)),
                          Constant(2.0))
        else:
            return BinaryOp("*", self, Variable(var))
    
    def evaluate(self, **values: float) -> float:
        if self.name not in values:
            from symbolic.errors import EvaluationError
            raise EvaluationError(f"Variable '{self.name}' not bound")
        return values[self.name]
    
    def to_string(self) -> str:
        return self.name
    
    def to_latex(self) -> str:
        return self.name
    
    def contains_var(self, var: str) -> bool:
        return self.name == var


@dataclass(frozen=True)
class BinaryOp(Expression):
    """Represents binary operations (e.g., +, -, *, /, ^)."""
    
    op: str
    left: Expression
    right: Expression
    
    def simplify(self) -> Expression:
        from symbolic.simplify import simplify_binary
        return simplify_binary(self)
    
    def differentiate(self, var: str) -> Expression:
        from symbolic.differentiate import differentiate_binary
        return differentiate_binary(self, var)
    
    def integrate(self, var: str) -> Expression:
        from symbolic.integrate import integrate_binary
        return integrate_binary(self, var)
    
    def evaluate(self, **values: float) -> float:
        left_val = self.left.evaluate(**values)
        right_val = self.right.evaluate(**values)
        
        if self.op == "+":
            return left_val + right_val
        elif self.op == "-":
            return left_val - right_val
        elif self.op == "*":
            return left_val * right_val
        elif self.op == "/":
            if right_val == 0:
                from symbolic.errors import EvaluationError
                raise EvaluationError("Division by zero")
            return left_val / right_val
        elif self.op == "^":
            return left_val ** right_val
        else:
            from symbolic.errors import EvaluationError
            raise EvaluationError(f"Unknown operator: {self.op}")
    
    def to_string(self) -> str:
        from symbolic.printer import binary_to_string
        return binary_to_string(self)
    
    def to_latex(self) -> str:
        from symbolic.printer import binary_to_latex
        return binary_to_latex(self)
    
    def contains_var(self, var: str) -> bool:
        return self.left.contains_var(var) or self.right.contains_var(var)


@dataclass(frozen=True)
class UnaryOp(Expression):
    """Represents unary operations (e.g., negation)."""
    
    op: str
    operand: Expression
    
    def simplify(self) -> Expression:
        from symbolic.simplify import simplify_unary
        return simplify_unary(self)
    
    def differentiate(self, var: str) -> Expression:
        from symbolic.differentiate import differentiate_unary
        return differentiate_unary(self, var)
    
    def integrate(self, var: str) -> Expression:
        from symbolic.integrate import integrate_unary
        return integrate_unary(self, var)
    
    def evaluate(self, **values: float) -> float:
        operand_val = self.operand.evaluate(**values)
        
        if self.op == "-":
            return -operand_val
        elif self.op == "+":
            return operand_val
        else:
            from symbolic.errors import EvaluationError
            raise EvaluationError(f"Unknown unary operator: {self.op}")
    
    def to_string(self) -> str:
        from symbolic.printer import unary_to_string
        return unary_to_string(self)
    
    def to_latex(self) -> str:
        from symbolic.printer import unary_to_latex
        return unary_to_latex(self)
    
    def contains_var(self, var: str) -> bool:
        return self.operand.contains_var(var)


@dataclass(frozen=True)
class Function(Expression):
    """Represents mathematical functions (e.g., sin, cos, exp, log)."""
    
    name: str
    arg: Expression
    
    def simplify(self) -> Expression:
        from symbolic.simplify import simplify_function
        return simplify_function(self)
    
    def differentiate(self, var: str) -> Expression:
        from symbolic.differentiate import differentiate_function
        return differentiate_function(self, var)
    
    def integrate(self, var: str) -> Expression:
        from symbolic.integrate import integrate_function
        return integrate_function(self, var)
    
    def evaluate(self, **values: float) -> float:
        arg_val = self.arg.evaluate(**values)
        
        if self.name == "sin":
            return math.sin(arg_val)
        elif self.name == "cos":
            return math.cos(arg_val)
        elif self.name == "tan":
            return math.tan(arg_val)
        elif self.name == "exp":
            return math.exp(arg_val)
        elif self.name == "log":
            if arg_val <= 0:
                from symbolic.errors import EvaluationError
                raise EvaluationError("Logarithm of non-positive number")
            return math.log(arg_val)
        elif self.name == "sqrt":
            if arg_val < 0:
                from symbolic.errors import EvaluationError
                raise EvaluationError("Square root of negative number")
            return math.sqrt(arg_val)
        elif self.name == "abs":
            return abs(arg_val)
        else:
            from symbolic.errors import EvaluationError
            raise EvaluationError(f"Unknown function: {self.name}")
    
    def to_string(self) -> str:
        from symbolic.printer import function_to_string
        return function_to_string(self)
    
    def to_latex(self) -> str:
        from symbolic.printer import function_to_latex
        return function_to_latex(self)
    
    def contains_var(self, var: str) -> bool:
        return self.arg.contains_var(var)
