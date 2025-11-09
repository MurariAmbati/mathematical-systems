"""
Test Suite for Math Art Generator

Comprehensive tests for core functionality.
"""

import pytest
import numpy as np
from core.parser import parse_equation, validate_expression
from core.evaluator import Evaluator
from core.coordinates import (
    cartesian_grid,
    polar_to_cartesian,
    parametric_to_points,
    normalize_points
)
from core.generators.spirograph import Spirograph
from core.generators.lissajous import Lissajous
from core.generators.attractors import LorenzAttractor, CliffordAttractor
from core.utils import SeededRNG, NoiseGenerator


class TestParser:
    """Test expression parsing."""
    
    def test_parse_simple_expression(self):
        """Test parsing a simple expression."""
        expr = parse_equation("sin(x) + cos(y)")
        assert expr is not None
        assert "x" in expr.variables
        assert "y" in expr.variables
    
    def test_parse_complex_expression(self):
        """Test parsing a complex expression."""
        expr = parse_equation("sin(x*y) + cos(x**2 - y**2)")
        assert expr is not None
    
    def test_parse_polar_expression(self):
        """Test parsing polar coordinates."""
        expr = parse_equation("sin(5*theta)")
        assert "theta" in expr.variables
    
    def test_invalid_variable(self):
        """Test that invalid variables raise error."""
        with pytest.raises(ValueError):
            parse_equation("sin(invalid_var)")
    
    def test_validate_expression(self):
        """Test expression validation."""
        result = validate_expression("sin(x) + cos(y)")
        assert result["valid"] is True
        assert "x" in result["variables"]
        assert "y" in result["variables"]
    
    def test_invalid_expression(self):
        """Test validation of invalid expression."""
        result = validate_expression("sin(")
        assert result["valid"] is False
        assert result["error"] is not None


class TestEvaluator:
    """Test expression evaluation."""
    
    def test_evaluate_simple(self):
        """Test simple evaluation."""
        evaluator = Evaluator("sin(x)")
        result = evaluator.evaluate(x=0)
        assert np.isclose(result, 0)
        
        result = evaluator.evaluate(x=np.pi/2)
        assert np.isclose(result, 1)
    
    def test_evaluate_array(self):
        """Test evaluation with arrays."""
        evaluator = Evaluator("x**2 + y**2")
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        result = evaluator.evaluate(x=x, y=y)
        expected = np.array([2, 8, 18])
        assert np.allclose(result, expected)
    
    def test_evaluate_grid(self):
        """Test grid evaluation."""
        evaluator = Evaluator("sin(x) + cos(y)")
        X, Y, Z = evaluator.evaluate_grid((-1, 1), (-1, 1), samples=10)
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert Z.shape == (10, 10)


class TestCoordinates:
    """Test coordinate transformations."""
    
    def test_cartesian_grid(self):
        """Test Cartesian grid generation."""
        X, Y = cartesian_grid((-1, 1), (-1, 1), samples=10)
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert X.min() >= -1 and X.max() <= 1
        assert Y.min() >= -1 and Y.max() <= 1
    
    def test_polar_to_cartesian(self):
        """Test polar to Cartesian conversion."""
        r = 1.0
        theta = 0.0
        x, y = polar_to_cartesian(r, theta)
        assert np.isclose(x, 1.0)
        assert np.isclose(y, 0.0)
        
        theta = np.pi/2
        x, y = polar_to_cartesian(r, theta)
        assert np.isclose(x, 0.0)
        assert np.isclose(y, 1.0)
    
    def test_parametric_to_points(self):
        """Test parametric curve generation."""
        fx = lambda t: np.sin(t)
        fy = lambda t: np.cos(t)
        points = parametric_to_points(fx, fy, (0, 2*np.pi), samples=100)
        assert points.shape == (100, 2)
    
    def test_normalize_points(self):
        """Test point normalization."""
        points = np.array([[0, 0], [10, 10], [5, 5]])
        normalized = normalize_points(points, target_range=(0, 1))
        assert normalized.min() >= 0
        assert normalized.max() <= 1


class TestGenerators:
    """Test art generators."""
    
    def test_spirograph_generation(self):
        """Test spirograph point generation."""
        spiro = Spirograph(R=5, r=3, a=2, samples=1000, seed=42)
        points = spiro.generate()
        assert points.shape[0] == 1000
        assert points.shape[1] == 2
    
    def test_lissajous_generation(self):
        """Test Lissajous curve generation."""
        lissajous = Lissajous(a=3, b=2, samples=1000, seed=42)
        points = lissajous.generate()
        assert points.shape[0] == 1000
        assert points.shape[1] == 2
    
    def test_lorenz_attractor(self):
        """Test Lorenz attractor generation."""
        lorenz = LorenzAttractor(iterations=1000, seed=42)
        points = lorenz.generate()
        assert points.shape[0] == 1000
        assert points.shape[1] == 3
    
    def test_clifford_attractor(self):
        """Test Clifford attractor generation."""
        clifford = CliffordAttractor(iterations=1000, seed=42)
        points = clifford.generate()
        assert points.shape[0] == 1000
        assert points.shape[1] == 2


class TestDeterminism:
    """Test deterministic behavior with seeds."""
    
    def test_spirograph_determinism(self):
        """Test spirograph reproducibility."""
        spiro1 = Spirograph(R=5, r=3, a=2, samples=100, seed=42)
        points1 = spiro1.generate()
        
        spiro2 = Spirograph(R=5, r=3, a=2, samples=100, seed=42)
        points2 = spiro2.generate()
        
        assert np.allclose(points1, points2)
    
    def test_attractor_determinism(self):
        """Test attractor reproducibility."""
        lorenz1 = LorenzAttractor(iterations=100, seed=42)
        points1 = lorenz1.generate()
        
        lorenz2 = LorenzAttractor(iterations=100, seed=42)
        points2 = lorenz2.generate()
        
        assert np.allclose(points1, points2)
    
    def test_rng_determinism(self):
        """Test RNG reproducibility."""
        rng1 = SeededRNG(seed=42)
        values1 = rng1.random(100)
        
        rng2 = SeededRNG(seed=42)
        values2 = rng2.random(100)
        
        assert np.allclose(values1, values2)
    
    def test_noise_determinism(self):
        """Test noise reproducibility."""
        noise1 = NoiseGenerator(seed=42)
        values1 = noise1.noise_2d(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        
        noise2 = NoiseGenerator(seed=42)
        values2 = noise2.noise_2d(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
        
        assert np.allclose(values1, values2)


class TestUtils:
    """Test utility functions."""
    
    def test_seeded_rng(self):
        """Test seeded RNG."""
        rng = SeededRNG(seed=42)
        values = rng.random(100)
        assert len(values) == 100
        assert values.min() >= 0
        assert values.max() < 1
    
    def test_noise_generator(self):
        """Test noise generation."""
        noise_gen = NoiseGenerator(seed=42)
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        values = noise_gen.noise_2d(x, y)
        assert len(values) == 10


class TestPerformance:
    """Test performance requirements."""
    
    def test_large_point_generation(self):
        """Test generation of 1M points completes quickly."""
        import time
        
        spiro = Spirograph(R=5, r=3, a=2, samples=1000000, seed=42)
        start = time.time()
        points = spiro.generate()
        elapsed = time.time() - start
        
        assert points.shape[0] == 1000000
        assert elapsed < 5.0  # Should complete in under 5 seconds
    
    def test_evaluation_performance(self):
        """Test evaluation of 1M points."""
        import time
        
        evaluator = Evaluator("sin(x) + cos(y)")
        x = np.random.randn(1000000)
        y = np.random.randn(1000000)
        
        start = time.time()
        result = evaluator.evaluate(x=x, y=y)
        elapsed = time.time() - start
        
        assert len(result) == 1000000
        assert elapsed < 2.0  # Should complete in under 2 seconds


class TestExport:
    """Test export functionality."""
    
    def test_config_export(self):
        """Test generator config export."""
        spiro = Spirograph(R=5, r=3, a=2, samples=100, seed=42)
        config = spiro.get_config()
        
        assert config["id"] == "spirograph"
        assert config["samples"] == 100
        assert config["seed"] == 42
        assert config["params"]["R"] == 5
    
    def test_config_roundtrip(self):
        """Test config save/load roundtrip."""
        spiro1 = Spirograph(R=5, r=3, a=2, samples=100, seed=42)
        config = spiro1.get_config()
        
        # Recreate from config
        spiro2 = Spirograph.from_config(config)
        
        points1 = spiro1.generate()
        points2 = spiro2.generate()
        
        assert np.allclose(points1, points2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
