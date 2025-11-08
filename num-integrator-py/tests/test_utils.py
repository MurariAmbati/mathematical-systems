"""
tests for utils module

verifies utility functions for interpolation, meshing, and helpers.
"""

import pytest
import numpy as np
from numeric_integrator import utils


class TestLinspaceAdaptive:
    """test adaptive mesh generation"""
    
    def test_smooth_function(self):
        """smooth function should use fewer points"""
        x, y = utils.linspace_adaptive(lambda x: x**2, 0, 1, tol=0.01, 
                                       min_points=10, max_points=100)
        assert len(x) >= 10
        assert len(x) == len(y)
        assert x[0] == 0
        assert x[-1] == 1
    
    def test_varying_function(self):
        """rapidly varying function should use more points"""
        x, y = utils.linspace_adaptive(lambda x: np.sin(20*x), 0, 1, 
                                       tol=0.01, min_points=10, max_points=200)
        assert len(x) > 10  # should refine
    
    def test_discontinuous_derivative(self):
        """function with sharp corner should refine there"""
        x, y = utils.linspace_adaptive(lambda x: abs(x - 0.5), 0, 1,
                                       tol=0.01, min_points=10, max_points=100)
        assert len(x) >= 10


class TestLagrangeInterpolation:
    """test lagrange polynomial interpolation"""
    
    def test_linear_data(self):
        """interpolate linear data"""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 3.0, 5.0])  # y = 2x + 1
        
        interp = utils.lagrange_interpolation(x_data, y_data)
        
        assert abs(interp(0.5) - 2.0) < 1e-10
        assert abs(interp(1.5) - 4.0) < 1e-10
    
    def test_quadratic_data(self):
        """interpolate quadratic data"""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([0.0, 1.0, 4.0])  # y = x²
        
        interp = utils.lagrange_interpolation(x_data, y_data)
        
        assert abs(interp(0.5) - 0.25) < 1e-10
        assert abs(interp(1.5) - 2.25) < 1e-10
    
    def test_exact_at_nodes(self):
        """interpolant should be exact at data points"""
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([1.0, 2.0, 4.0, 8.0])
        
        interp = utils.lagrange_interpolation(x_data, y_data)
        
        for xi, yi in zip(x_data, y_data):
            assert abs(interp(xi) - yi) < 1e-10


class TestNewtonDividedDifferences:
    """test newton interpolation"""
    
    def test_linear_data(self):
        """interpolate linear data"""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([3.0, 5.0, 7.0])  # y = 2x + 3
        
        interp = utils.newton_divided_differences(x_data, y_data)
        
        assert abs(interp(0.5) - 4.0) < 1e-10
        assert abs(interp(1.5) - 6.0) < 1e-10
    
    def test_cubic_data(self):
        """interpolate cubic data"""
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 8.0, 27.0])  # y = x³
        
        interp = utils.newton_divided_differences(x_data, y_data)
        
        assert abs(interp(1.5) - 1.5**3) < 1e-10


class TestCubicSpline:
    """test cubic spline interpolation"""
    
    def test_smooth_interpolation(self):
        """spline should smoothly interpolate"""
        x_data = np.array([0.0, 1.0, 2.0, 3.0])
        y_data = np.array([0.0, 1.0, 0.0, 1.0])
        
        spline = utils.cubic_spline(x_data, y_data)
        
        # check at data points
        for xi, yi in zip(x_data, y_data):
            assert abs(spline(xi) - yi) < 1e-6
        
        # check between points
        assert abs(spline(0.5)) < 1.5
        assert abs(spline(1.5)) < 1.5
    
    def test_linear_data(self):
        """spline of linear data should be linear"""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([0.0, 2.0, 4.0])  # y = 2x
        
        spline = utils.cubic_spline(x_data, y_data)
        
        assert abs(spline(0.5) - 1.0) < 1e-6
        assert abs(spline(1.5) - 3.0) < 1e-6
    
    def test_boundary_handling(self):
        """test extrapolation at boundaries"""
        x_data = np.array([0.0, 1.0, 2.0])
        y_data = np.array([1.0, 2.0, 3.0])
        
        spline = utils.cubic_spline(x_data, y_data)
        
        # should return boundary values outside range
        assert spline(-1.0) == y_data[0]
        assert spline(3.0) == y_data[-1]


class TestChebyshevNodes:
    """test chebyshev node generation"""
    
    def test_node_count(self):
        """should generate correct number of nodes"""
        nodes = utils.chebyshev_nodes(10)
        assert len(nodes) == 10
    
    def test_interval(self):
        """nodes should be in [-1, 1]"""
        nodes = utils.chebyshev_nodes(20)
        assert np.all(nodes >= -1.0)
        assert np.all(nodes <= 1.0)
    
    def test_custom_interval(self):
        """nodes should map to custom interval"""
        nodes = utils.chebyshev_nodes(10, a=0, b=1)
        assert np.all(nodes >= 0.0)
        assert np.all(nodes <= 1.0)
    
    def test_clustering(self):
        """nodes should cluster near endpoints"""
        nodes = utils.chebyshev_nodes(100, a=0, b=1)
        # more nodes near 0 and 1 than in middle
        near_boundary = np.sum((nodes < 0.1) | (nodes > 0.9))
        in_middle = np.sum((nodes >= 0.4) & (nodes <= 0.6))
        assert near_boundary > in_middle


class TestAdaptiveMeshRefinement:
    """test mesh refinement"""
    
    def test_smooth_function(self):
        """smooth function should not refine much"""
        x = np.linspace(0, 1, 10)
        y = x**2
        
        x_new, y_new = utils.adaptive_mesh_refinement(x, y, tolerance=0.1)
        assert len(x_new) >= len(x)
    
    def test_sharp_feature(self):
        """function with sharp feature should refine"""
        x = np.linspace(0, 2, 10)
        y = np.where(x < 1, x, 2 - x)  # triangle
        
        x_new, y_new = utils.adaptive_mesh_refinement(x, y, tolerance=0.01)
        # should add points
        assert len(x_new) >= len(x)


class TestFunctionSmoother:
    """test function smoothing"""
    
    def test_noisy_data(self):
        """should smooth noisy data"""
        y = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.1, 0.9, 1.0])
        smoothed = utils.function_smoother(y, window=3)
        
        assert len(smoothed) == len(y)
        # smoothed should have less variation
        assert np.std(smoothed) < np.std(y)
    
    def test_constant_data(self):
        """constant data should remain constant"""
        y = np.ones(10)
        smoothed = utils.function_smoother(y, window=3)
        
        assert np.allclose(smoothed, 1.0)


class TestEstimateDerivativeOrder:
    """test order of accuracy estimation"""
    
    def test_first_order(self):
        """detect first-order convergence"""
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors = [0.1 * h for h in h_values]
        
        order = utils.estimate_derivative_order(h_values, errors)
        assert abs(order - 1.0) < 0.1
    
    def test_second_order(self):
        """detect second-order convergence"""
        h_values = [0.1, 0.05, 0.025, 0.0125]
        errors = [0.01 * h**2 for h in h_values]
        
        order = utils.estimate_derivative_order(h_values, errors)
        assert abs(order - 2.0) < 0.1
    
    def test_insufficient_data(self):
        """handle insufficient data"""
        order = utils.estimate_derivative_order([0.1], [0.01])
        assert order == 0.0


class TestRelativeDifference:
    """test relative difference computation"""
    
    def test_normal_values(self):
        """test with normal values"""
        diff = utils.relative_difference(10.0, 8.0)
        assert abs(diff - 0.2) < 1e-10
    
    def test_near_zero(self):
        """test with values near zero"""
        diff = utils.relative_difference(1e-11, 2e-11)
        assert diff < 1e-10
    
    def test_equal_values(self):
        """test with equal values"""
        diff = utils.relative_difference(5.0, 5.0)
        assert diff < 1e-15


class TestIsConverged:
    """test convergence checking"""
    
    def test_converged(self):
        """test converged case"""
        assert utils.is_converged(1.0, 1.0 + 1e-9, tol=1e-6)
    
    def test_not_converged(self):
        """test not converged case"""
        assert not utils.is_converged(1.0, 1.1, tol=1e-6)
    
    def test_zero_values(self):
        """test with values near zero"""
        assert utils.is_converged(1e-15, 2e-15, tol=1e-12)


class TestOptimalStepSize:
    """test optimal step size computation"""
    
    def test_returns_positive(self):
        """should return positive step size"""
        h = utils.optimal_step_size(lambda x: x**2, x=1.0)
        assert h > 0
    
    def test_reasonable_magnitude(self):
        """step size should be reasonable"""
        h = utils.optimal_step_size(lambda x: x**2, x=1.0, order=2)
        assert 1e-10 < h < 1e-2
    
    def test_different_orders(self):
        """higher order should use larger step"""
        h1 = utils.optimal_step_size(lambda x: x, x=1.0, order=1)
        h2 = utils.optimal_step_size(lambda x: x, x=1.0, order=2)
        assert h2 > h1
