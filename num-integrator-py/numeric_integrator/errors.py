"""
custom exceptions for numerical methods
"""


class NumericalError(Exception):
    """base exception for all numerical computation errors"""
    pass


class IntegrationError(NumericalError):
    """raised when integration fails or produces invalid results"""
    pass


class DifferentiationError(NumericalError):
    """raised when differentiation fails or step size is invalid"""
    pass


class ODEError(NumericalError):
    """raised when ode solver fails or encounters instability"""
    pass


class ConvergenceError(NumericalError):
    """raised when iterative method fails to converge"""
    pass


class StabilityError(NumericalError):
    """raised when stability criteria are violated"""
    pass
