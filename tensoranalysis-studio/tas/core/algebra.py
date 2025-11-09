"""
Tensor algebra operations: symmetrize, antisymmetrize, trace, etc.
"""

from typing import Sequence
import numpy as np
from itertools import permutations

from tas.core.tensor import Tensor
from tas.core.utils import normalize_axis


def symmetrize(tensor: Tensor, axes: Sequence[int]) -> Tensor:
    """
    Symmetrize a tensor over specified axes.
    
    Computes the average over all permutations of the specified axes.
    
    Args:
        tensor: Input tensor
        axes: Axes to symmetrize over
        
    Returns:
        Symmetrized tensor
        
    Examples:
        >>> T = Tensor(np.random.rand(3, 3, 3), indices=("^i", "^j", "^k"))
        >>> S = symmetrize(T, axes=(0, 1))  # Symmetric in first two indices
    """
    axes_tuple = tuple(normalize_axis(ax, tensor.ndim) for ax in axes)
    
    if len(axes_tuple) < 2:
        return tensor
    
    # Generate all permutations
    perms = list(permutations(axes_tuple))
    
    # Accumulate symmetrized result (use float to avoid casting issues)
    result = np.zeros_like(tensor.data, dtype=float)
    
    for perm in perms:
        # Build full permutation (keep other axes in place)
        full_perm = list(range(tensor.ndim))
        for i, ax in enumerate(axes_tuple):
            full_perm[ax] = perm[i]
        
        result += np.transpose(tensor.data, full_perm)
    
    result /= len(perms)
    
    return Tensor(
        data=result,
        indices=tensor.indices,
        name=f"sym({tensor.name or 'T'})",
        backend=tensor.backend,
        meta=tensor.meta
    )


def antisymmetrize(tensor: Tensor, axes: Sequence[int]) -> Tensor:
    """
    Antisymmetrize (alternating) a tensor over specified axes.
    
    Computes the alternating sum over all permutations of the specified axes,
    with sign determined by permutation parity.
    
    Args:
        tensor: Input tensor
        axes: Axes to antisymmetrize over
        
    Returns:
        Antisymmetrized tensor
        
    Examples:
        >>> T = Tensor(np.random.rand(3, 3), indices=("^i", "^j"))
        >>> A = antisymmetrize(T, axes=(0, 1))  # Antisymmetric matrix
    """
    axes_tuple = tuple(normalize_axis(ax, tensor.ndim) for ax in axes)
    
    if len(axes_tuple) < 2:
        return tensor
    
    # Generate all permutations with their signs
    from itertools import permutations as iter_perms
    
    def permutation_sign(perm: Sequence[int]) -> int:
        """Compute sign of permutation (Levi-Civita)."""
        n = len(perm)
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if perm[i] > perm[j]:
                    sign *= -1
        return sign
    
    perms = list(iter_perms(range(len(axes_tuple))))
    
    # Accumulate antisymmetrized result (use float to avoid casting issues)
    result = np.zeros_like(tensor.data, dtype=float)
    
    for perm in perms:
        # Build full permutation
        full_perm = list(range(tensor.ndim))
        for i, ax in enumerate(axes_tuple):
            full_perm[ax] = axes_tuple[perm[i]]
        
        sign = permutation_sign(perm)
        result += sign * np.transpose(tensor.data, full_perm)
    
    result /= len(perms)
    
    return Tensor(
        data=result,
        indices=tensor.indices,
        name=f"antisym({tensor.name or 'T'})",
        backend=tensor.backend,
        meta=tensor.meta
    )


def trace(tensor: Tensor, axis1: int = 0, axis2: int = 1) -> Tensor:
    """
    Compute trace over two axes of a tensor.
    
    Sums over diagonal elements where the two axes are equal.
    The two axes must have the same dimension.
    
    Args:
        tensor: Input tensor
        axis1: First axis to trace over
        axis2: Second axis to trace over
        
    Returns:
        Tensor with rank reduced by 2
        
    Raises:
        ValueError: If axes have different dimensions
        
    Examples:
        >>> M = Tensor(np.random.rand(3, 3), indices=("^i", "_i"))
        >>> tr = trace(M, 0, 1)  # Scalar
    """
    ax1 = normalize_axis(axis1, tensor.ndim)
    ax2 = normalize_axis(axis2, tensor.ndim)
    
    if ax1 == ax2:
        raise ValueError("Cannot trace over the same axis twice")
    
    if tensor.shape[ax1] != tensor.shape[ax2]:
        raise ValueError(
            f"Cannot trace over axes with different dimensions: "
            f"{tensor.shape[ax1]} vs {tensor.shape[ax2]}"
        )
    
    # Use numpy trace
    result_data = np.trace(tensor.data, axis1=ax1, axis2=ax2)
    
    # Determine remaining indices
    remaining_indices = [
        tensor.indices[i] for i in range(tensor.ndim) 
        if i != ax1 and i != ax2
    ]
    
    # If result is scalar, wrap in 0-d array
    if not remaining_indices:
        result_data = np.asarray(result_data)
    
    return Tensor(
        data=result_data,
        indices=tuple(remaining_indices),
        name=f"trace({tensor.name or 'T'})",
        backend=tensor.backend,
        meta=tensor.meta
    )


def wedge(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Wedge (exterior) product of two antisymmetric tensors.
    
    Computes the antisymmetrized tensor product, useful for differential forms.
    
    Args:
        tensor1: First antisymmetric tensor
        tensor2: Second antisymmetric tensor
        
    Returns:
        Wedge product (antisymmetric tensor)
    """
    from tas.core.einsum_parser import outer
    
    # Compute outer product
    product = outer(tensor1, tensor2)
    
    # Antisymmetrize over all indices
    all_axes = tuple(range(product.ndim))
    return antisymmetrize(product, all_axes)


def tensor_product(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Direct tensor product (⊗) of two tensors.
    
    Alias for outer product. Result has rank = rank1 + rank2.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Tensor product
    """
    from tas.core.einsum_parser import outer
    return outer(tensor1, tensor2)
