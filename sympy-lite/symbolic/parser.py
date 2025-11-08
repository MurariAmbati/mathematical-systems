"""
Expression parser for converting string mathematical expressions to AST.

Implements a recursive descent parser with proper operator precedence:
1. Parentheses
2. Functions (sin, cos, exp, log, etc.)
3. Exponentiation (^) - right associative
4. Unary operators (+, -)
5. Multiplication and division (*, /)
6. Addition and subtraction (+, -)
"""

import re
from typing import List, Optional
from dataclasses import dataclass
import math

from symbolic.ast import Expression, Constant, Variable, BinaryOp, UnaryOp, Function
from symbolic.errors import ParseError


@dataclass
class Token:
    """Represents a token in the input string."""
    type: str  # 'NUMBER', 'VARIABLE', 'OPERATOR', 'FUNCTION', 'LPAREN', 'RPAREN', 'COMMA'
    value: str
    position: int


class Tokenizer:
    """Tokenizes mathematical expression strings."""
    
    # Regular expression patterns
    PATTERNS = [
        ('NUMBER', r'\d+\.?\d*'),
        ('FUNCTION', r'(sin|cos|tan|exp|log|sqrt|abs)\b'),
        ('VARIABLE', r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('OPERATOR', r'[\+\-\*/\^]'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('COMMA', r','),
        ('WHITESPACE', r'\s+'),
    ]
    
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.tokens: List[Token] = []
        self._tokenize()
    
    def _tokenize(self) -> None:
        """Convert input string into list of tokens."""
        while self.position < len(self.text):
            matched = False
            
            for token_type, pattern in self.PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.text, self.position)
                
                if match:
                    value = match.group(0)
                    
                    # Skip whitespace
                    if token_type != 'WHITESPACE':
                        self.tokens.append(Token(token_type, value, self.position))
                    
                    self.position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise ParseError(
                    f"Invalid character '{self.text[self.position]}' at position {self.position}",
                    self.position
                )


class Parser:
    """Recursive descent parser for mathematical expressions."""
    
    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.position = 0
    
    def current_token(self) -> Optional[Token]:
        """Get current token without consuming it."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
    
    def consume(self, expected_type: Optional[str] = None) -> Token:
        """Consume and return current token."""
        if self.position >= len(self.tokens):
            raise ParseError("Unexpected end of expression")
        
        token = self.tokens[self.position]
        
        if expected_type and token.type != expected_type:
            raise ParseError(
                f"Expected {expected_type} but got {token.type} at position {token.position}",
                token.position
            )
        
        self.position += 1
        return token
    
    def parse(self) -> Expression:
        """Parse the token stream into an expression tree."""
        expr = self.parse_additive()
        
        # Ensure all tokens consumed
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            raise ParseError(f"Unexpected token '{token.value}' at position {token.position}", token.position)
        
        return expr
    
    def parse_additive(self) -> Expression:
        """Parse addition and subtraction (lowest precedence)."""
        left = self.parse_multiplicative()
        
        while True:
            token = self.current_token()
            if token and token.type == 'OPERATOR' and token.value in ['+', '-']:
                op = self.consume().value
                right = self.parse_multiplicative()
                left = BinaryOp(op, left, right)
            else:
                break
        
        return left
    
    def parse_multiplicative(self) -> Expression:
        """Parse multiplication and division."""
        left = self.parse_unary()
        
        while True:
            token = self.current_token()
            if token and token.type == 'OPERATOR' and token.value in ['*', '/']:
                op = self.consume().value
                right = self.parse_unary()
                left = BinaryOp(op, left, right)
            else:
                break
        
        return left
    
    def parse_unary(self) -> Expression:
        """Parse unary operators."""
        token = self.current_token()
        
        if token and token.type == 'OPERATOR' and token.value in ['+', '-']:
            op = self.consume().value
            operand = self.parse_unary()
            return UnaryOp(op, operand)
        
        return self.parse_power()
    
    def parse_power(self) -> Expression:
        """Parse exponentiation (right associative)."""
        left = self.parse_primary()
        
        token = self.current_token()
        if token and token.type == 'OPERATOR' and token.value == '^':
            self.consume()
            # Right associative: a^b^c = a^(b^c)
            right = self.parse_power()
            return BinaryOp('^', left, right)
        
        return left
    
    def parse_primary(self) -> Expression:
        """Parse primary expressions: numbers, variables, functions, parenthesized expressions."""
        token = self.current_token()
        
        if not token:
            raise ParseError("Unexpected end of expression")
        
        # Numbers
        if token.type == 'NUMBER':
            self.consume()
            value = float(token.value)
            return Constant(value)
        
        # Variables (including special constants)
        if token.type == 'VARIABLE':
            self.consume()
            # Handle special constants
            if token.value == 'pi':
                return Constant(math.pi)
            elif token.value == 'e':
                return Constant(math.e)
            else:
                return Variable(token.value)
        
        # Functions
        if token.type == 'FUNCTION':
            func_name = self.consume().value
            self.consume('LPAREN')
            arg = self.parse_additive()
            self.consume('RPAREN')
            return Function(func_name, arg)
        
        # Parenthesized expressions
        if token.type == 'LPAREN':
            self.consume()
            expr = self.parse_additive()
            self.consume('RPAREN')
            return expr
        
        raise ParseError(f"Unexpected token '{token.value}' at position {token.position}", token.position)


def parse(expression: str) -> Expression:
    """
    Parse a mathematical expression string into an AST.
    
    Args:
        expression: String representation of a mathematical expression
        
    Returns:
        Expression: Root node of the AST
        
    Raises:
        ParseError: If the expression is invalid
        
    Examples:
        >>> expr = parse("x^2 + 2*x + 1")
        >>> expr = parse("sin(x) * exp(x)")
        >>> expr = parse("(x + 1)^3 / (2*x)")
    """
    if not expression or not expression.strip():
        raise ParseError("Empty expression")
    
    tokenizer = Tokenizer(expression)
    parser = Parser(tokenizer.tokens)
    return parser.parse()
