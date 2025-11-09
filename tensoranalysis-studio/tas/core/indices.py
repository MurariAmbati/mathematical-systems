"""
Index labels and metadata for tensor indices.

Indices track covariant/contravariant variance and provide
unique labels for Einstein summation notation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Union


IndexVariance = Literal["up", "down"]


@dataclass(frozen=True)
class Index:
    """
    Immutable index label with variance information.
    
    Represents a single tensor index with:
    - A unique name/label (e.g., 'i', 'mu', 'alpha')
    - Variance: 'up' (contravariant, superscript) or 'down' (covariant, subscript)
    - Optional dimension and basis information
    
    Examples:
        >>> i_down = Index("i", variance="down")  # Covariant index _i
        >>> j_up = Index("j", variance="up")      # Contravariant index ^j
        >>> mu = Index("mu", variance="down", dimension=4)  # Spacetime index
    """
    
    name: str
    variance: IndexVariance
    dimension: Optional[int] = None
    basis: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate index parameters."""
        if not self.name:
            raise ValueError("Index name cannot be empty")
        
        if self.variance not in ("up", "down"):
            raise ValueError(f"Invalid variance: {self.variance}. Must be 'up' or 'down'.")
        
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {self.dimension}")
    
    def __str__(self) -> str:
        """String representation using index notation."""
        prefix = "^" if self.variance == "up" else "_"
        return f"{prefix}{self.name}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        parts = [f"name={self.name!r}", f"variance={self.variance!r}"]
        if self.dimension is not None:
            parts.append(f"dimension={self.dimension}")
        if self.basis is not None:
            parts.append(f"basis={self.basis!r}")
        return f"Index({', '.join(parts)})"
    
    def raise_index(self) -> "Index":
        """
        Return a new index with variance changed to 'up' (contravariant).
        
        Returns:
            New Index with variance='up'
        """
        if self.variance == "up":
            return self
        return Index(
            name=self.name,
            variance="up",
            dimension=self.dimension,
            basis=self.basis
        )
    
    def lower_index(self) -> "Index":
        """
        Return a new index with variance changed to 'down' (covariant).
        
        Returns:
            New Index with variance='down'
        """
        if self.variance == "down":
            return self
        return Index(
            name=self.name,
            variance="down",
            dimension=self.dimension,
            basis=self.basis
        )
    
    def flip_variance(self) -> "Index":
        """
        Return a new index with opposite variance.
        
        Returns:
            New Index with flipped variance
        """
        new_variance: IndexVariance = "down" if self.variance == "up" else "up"
        return Index(
            name=self.name,
            variance=new_variance,
            dimension=self.dimension,
            basis=self.basis
        )
    
    def matches(self, other: "Index", check_variance: bool = True) -> bool:
        """
        Check if this index matches another (for contraction compatibility).
        
        Args:
            other: Another Index to compare
            check_variance: If True, also check that variances are opposite
            
        Returns:
            True if indices match (and have opposite variance if check_variance=True)
        """
        if self.name != other.name:
            return False
        
        if self.dimension is not None and other.dimension is not None:
            if self.dimension != other.dimension:
                return False
        
        if check_variance:
            return self.variance != other.variance
        
        return True


def parse_index_string(s: str) -> Index:
    """
    Parse an index string like '^i', '_j', 'i' into an Index object.
    
    Convention:
    - '^i' or 'i^' -> contravariant (up)
    - '_i' or 'i_' -> covariant (down)
    - 'i' without prefix -> defaults to covariant (down)
    
    Args:
        s: Index string
        
    Returns:
        Parsed Index object
        
    Raises:
        ValueError: If string format is invalid
        
    Examples:
        >>> parse_index_string("^i")
        Index(name='i', variance='up')
        >>> parse_index_string("_mu")
        Index(name='mu', variance='down')
        >>> parse_index_string("j")
        Index(name='j', variance='down')
    """
    s = s.strip()
    
    if not s:
        raise ValueError("Cannot parse empty index string")
    
    # Check for prefix notation: ^i or _i
    if s.startswith("^"):
        return Index(name=s[1:], variance="up")
    elif s.startswith("_"):
        return Index(name=s[1:], variance="down")
    
    # Check for suffix notation: i^ or i_
    if s.endswith("^"):
        return Index(name=s[:-1], variance="up")
    elif s.endswith("_"):
        return Index(name=s[:-1], variance="down")
    
    # Default: no marker means covariant (down)
    return Index(name=s, variance="down")


def parse_index_tuple(indices: tuple[Union[str, Index], ...]) -> tuple[Index, ...]:
    """
    Parse a tuple of index strings or Index objects into Index objects.
    
    Args:
        indices: Tuple of strings or Index objects
        
    Returns:
        Tuple of Index objects
        
    Examples:
        >>> parse_index_tuple(("^i", "_j"))
        (Index(name='i', variance='up'), Index(name='j', variance='down'))
    """
    result: list[Index] = []
    for idx in indices:
        if isinstance(idx, Index):
            result.append(idx)
        elif isinstance(idx, str):
            result.append(parse_index_string(idx))
        else:
            raise TypeError(f"Expected str or Index, got {type(idx)}")
    return tuple(result)
