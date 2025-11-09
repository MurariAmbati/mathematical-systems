"""
Tests for expression parser module.
"""

import pytest
import numpy as np
from math_music_engine.parser import ExpressionParser, parse_expression, expression_to_function


class TestExpressionParser:
    """Test suite for ExpressionParser."""
    
    def test_parse_simple_expression(self):
        """Test parsing a simple expression."""
        parser = ExpressionParser()
        expr, metadata = parser.parse("sin(t)")
        
        assert metadata['variables'] == ['t']
        assert metadata['contains_trig']
        assert 'sin' in str(expr)
    
    def test_parse_complex_expression(self):
        """Test parsing a complex expression."""
        parser = ExpressionParser()
        expr, metadata = parser.parse("2*pi*440*t + cos(3*t)")
        
        assert 't' in metadata['variables']
        assert metadata['contains_trig']
    
    def test_to_numpy_function(self):
        """Test conversion to NumPy function."""
        parser = ExpressionParser()
        func, metadata = parser.to_numpy_function("sin(2*pi*t)")
        
        # Test function evaluation
        t = np.array([0, 0.25, 0.5, 0.75, 1.0])
        result = func(t)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(t)
        assert np.allclose(result[0], 0, atol=1e-10)  # sin(0) = 0
    
    def test_differentiate(self):
        """Test differentiation."""
        parser = ExpressionParser()
        derivative = parser.differentiate("t**2", "t")
        
        assert "2*t" in derivative
    
    def test_simplify(self):
        """Test expression simplification."""
        parser = ExpressionParser()
        simplified = parser.simplify("x + x")
        
        assert simplified == "2*x"
    
    def test_substitute(self):
        """Test value substitution."""
        parser = ExpressionParser()
        result = parser.substitute("x**2 + y", {'x': 3, 'y': 4})
        
        assert result == 13.0
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        expr, metadata = parse_expression("cos(t)")
        assert 't' in metadata['variables']
        
        func = expression_to_function("sin(t)")
        result = func(np.array([0]))
        assert np.allclose(result[0], 0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])
