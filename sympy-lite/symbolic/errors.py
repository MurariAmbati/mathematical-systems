"""Custom exception classes for the symbolic algebra engine."""


class SymbolicError(Exception):
    """Base exception for all symbolic solver errors."""
    pass


class ParseError(SymbolicError):
    """Raised when parsing an expression fails."""
    
    def __init__(self, message: str, position: int = -1) -> None:
        self.position = position
        super().__init__(message)


class EvaluationError(SymbolicError):
    """Raised when evaluation of an expression fails."""
    pass


class IntegrationError(SymbolicError):
    """Raised when integration cannot be performed."""
    pass


class SimplificationError(SymbolicError):
    """Raised when simplification encounters an error."""
    pass
