"""
Einstein notation parser and evaluator for tensor contractions.

Supports expressions like:
- "A^i_j B^j_k" - implicit summation over j
- "T^ab S_bc" - contract over b
- "g_ij T^ij" - full contraction
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from tas.core.tensor import Tensor
from tas.core.indices import Index


class EinsumExpression:
    """
    Parsed Einstein summation expression.
    
    Represents a tensor contraction expression parsed from string notation.
    Tracks which indices are contracted (appear twice) and builds an
    execution plan for efficient evaluation.
    """
    
    def __init__(self, expression: str):
        """
        Parse an Einstein notation expression.
        
        Args:
            expression: String like "A^i_j B^j_k" or "T^ab S_bc"
            
        Examples:
            >>> expr = EinsumExpression("A^i_j B^j_k")
            >>> expr.tensor_names
            ['A', 'B']
        """
        self.raw_expression = expression.strip()
        self.tensor_specs: List[Tuple[str, List[Tuple[str, str]]]] = []
        self.all_indices: List[Tuple[str, str]] = []  # (name, variance)
        self.contracted_indices: Set[str] = set()
        self.free_indices: List[Tuple[str, str]] = []
        
        self._parse()
        self._analyze_contractions()
    
    def _parse(self) -> None:
        """Parse the expression into tensor names and their indices."""
        # Pattern: TensorName followed by indices with ^ or _
        # Supports: A^i_j, T_ij, S^{ab}, etc.
        
        # Split by whitespace to get individual tensor terms
        terms = self.raw_expression.split()
        
        for term in terms:
            # Match tensor name and indices
            # Pattern: name followed by index specifications
            match = re.match(r'([A-Za-z][A-Za-z0-9]*)(.*)', term)
            if not match:
                continue
            
            tensor_name = match.group(1)
            indices_str = match.group(2)
            
            # Parse indices from the string
            indices = self._parse_indices(indices_str)
            
            if indices:  # Only add if there are indices
                self.tensor_specs.append((tensor_name, indices))
                self.all_indices.extend(indices)
    
    def _parse_indices(self, s: str) -> List[Tuple[str, str]]:
        """
        Parse index string into (name, variance) pairs.
        
        Supports:
        - ^i_j -> [('i', 'up'), ('j', 'down')]
        - ^{ij} -> [('i', 'up'), ('j', 'up')]
        - _ab -> [('a', 'down'), ('b', 'down')]
        """
        indices: List[Tuple[str, str]] = []
        i = 0
        
        while i < len(s):
            char = s[i]
            
            if char in ('^', '_'):
                variance = 'up' if char == '^' else 'down'
                i += 1
                
                # Check for braces
                if i < len(s) and s[i] == '{':
                    i += 1
                    # Find closing brace
                    start = i
                    while i < len(s) and s[i] != '}':
                        i += 1
                    index_names = s[start:i]
                    i += 1  # skip }
                    
                    # Each character in braces is an index
                    for name in index_names:
                        if name.isalnum():
                            indices.append((name, variance))
                else:
                    # Single character index or multi-char word
                    start = i
                    while i < len(s) and s[i].isalnum():
                        i += 1
                    
                    if start < i:
                        index_name = s[start:i]
                        # For multi-char, treat as single index
                        # For single char, treat each as separate
                        if len(index_name) == 1 or index_name.isalpha():
                            indices.append((index_name, variance))
                        else:
                            # Multi-char: split into individual indices
                            for name in index_name:
                                indices.append((name, variance))
            else:
                i += 1
        
        return indices
    
    def _analyze_contractions(self) -> None:
        """Determine which indices are contracted (summed over)."""
        index_count: Dict[str, List[str]] = {}  # name -> list of variances
        
        for name, variance in self.all_indices:
            if name not in index_count:
                index_count[name] = []
            index_count[name].append(variance)
        
        # Find contracted indices (appear exactly twice, once up and once down)
        for name, variances in index_count.items():
            if len(variances) == 2:
                # Standard Einstein convention: one up, one down
                if 'up' in variances and 'down' in variances:
                    self.contracted_indices.add(name)
            elif len(variances) > 2:
                # Appears more than twice - this is an error or needs special handling
                raise ValueError(
                    f"Index '{name}' appears {len(variances)} times. "
                    "Einstein notation requires at most 2 occurrences."
                )
        
        # Free indices are those not contracted
        seen_free: Set[str] = set()
        for name, variance in self.all_indices:
            if name not in self.contracted_indices and name not in seen_free:
                self.free_indices.append((name, variance))
                seen_free.add(name)
    
    @property
    def tensor_names(self) -> List[str]:
        """Get list of tensor names in the expression."""
        return [name for name, _ in self.tensor_specs]


def einsum_eval(expression: str, **tensors: Tensor) -> Tensor:
    """
    Evaluate an Einstein summation expression with given tensors.
    
    This is the main entry point for Einstein notation tensor contractions.
    Parses the expression, validates tensor compatibility, and performs
    the contraction using numpy's einsum.
    
    Args:
        expression: Einstein notation string (e.g., "A^i_j B^j_k")
        **tensors: Named tensors referenced in the expression
        
    Returns:
        Result tensor with appropriate indices
        
    Raises:
        ValueError: If expression is invalid or tensors incompatible
        
    Examples:
        >>> A = Tensor(np.random.rand(3, 3), indices=("^i", "_j"))
        >>> B = Tensor(np.random.rand(3, 3), indices=("^j", "_k"))
        >>> C = einsum_eval("A^i_j B^j_k", A=A, B=B)
        >>> C.shape
        (3, 3)
    """
    # Parse expression
    parsed = EinsumExpression(expression)
    
    # Check all tensors are provided
    for name in parsed.tensor_names:
        if name not in tensors:
            raise ValueError(f"Tensor '{name}' referenced in expression but not provided")
    
    # Build numpy einsum notation
    # For each tensor, build its index string
    einsum_inputs: List[str] = []
    einsum_tensors: List[np.ndarray] = []
    
    # Track index positions for result
    index_to_char: Dict[str, str] = {}
    char_pool = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    char_idx = 0
    
    # Assign characters to indices
    for name, variance in parsed.all_indices:
        if name not in index_to_char:
            if char_idx >= len(char_pool):
                raise ValueError("Too many unique indices for einsum")
            index_to_char[name] = char_pool[char_idx]
            char_idx += 1
    
    # Build einsum strings for each tensor
    for tensor_name, indices_spec in parsed.tensor_specs:
        tensor = tensors[tensor_name]
        
        # Validate index count matches tensor rank
        if len(indices_spec) != tensor.ndim:
            raise ValueError(
                f"Tensor '{tensor_name}' has rank {tensor.ndim} but "
                f"expression specifies {len(indices_spec)} indices"
            )
        
        # Build einsum string for this tensor
        einsum_str = ''.join(index_to_char[name] for name, _ in indices_spec)
        einsum_inputs.append(einsum_str)
        einsum_tensors.append(tensor.data)
    
    # Build output string (free indices only)
    output_chars = [index_to_char[name] for name, _ in parsed.free_indices]
    output_str = ''.join(output_chars)
    
    # Construct full einsum expression
    full_einsum = ','.join(einsum_inputs) + '->' + output_str
    
    # Perform contraction
    result_data = np.einsum(full_einsum, *einsum_tensors)
    
    # Build result indices
    result_indices = [Index(name, variance) for name, variance in parsed.free_indices]
    
    # Determine result name
    if len(tensors) == 1:
        result_name = f"contract({list(tensors.keys())[0]})"
    else:
        result_name = f"contract({','.join(tensors.keys())})"
    
    return Tensor(
        data=result_data,
        indices=tuple(result_indices),
        name=result_name
    )


def contract(tensor1: Tensor, tensor2: Tensor, 
             axes1: Tuple[int, ...], axes2: Tuple[int, ...]) -> Tensor:
    """
    Contract two tensors along specified axes.
    
    This is a lower-level contraction function that doesn't use Einstein notation.
    Directly specifies which axes to contract.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        axes1: Axes in tensor1 to contract
        axes2: Axes in tensor2 to contract
        
    Returns:
        Contracted tensor
        
    Raises:
        ValueError: If axes are incompatible
        
    Examples:
        >>> A = Tensor(np.random.rand(3, 4), indices=("^i", "_j"))
        >>> B = Tensor(np.random.rand(4, 5), indices=("^j", "_k"))
        >>> C = contract(A, B, axes1=(1,), axes2=(0,))
    """
    if len(axes1) != len(axes2):
        raise ValueError("Must contract same number of axes from each tensor")
    
    # Validate axes
    for ax1 in axes1:
        if ax1 < 0 or ax1 >= tensor1.ndim:
            raise ValueError(f"Invalid axis {ax1} for tensor1 with rank {tensor1.ndim}")
    
    for ax2 in axes2:
        if ax2 < 0 or ax2 >= tensor2.ndim:
            raise ValueError(f"Invalid axis {ax2} for tensor2 with rank {tensor2.ndim}")
    
    # Check dimension compatibility
    for ax1, ax2 in zip(axes1, axes2):
        if tensor1.shape[ax1] != tensor2.shape[ax2]:
            raise ValueError(
                f"Cannot contract axes with different dimensions: "
                f"{tensor1.shape[ax1]} vs {tensor2.shape[ax2]}"
            )
    
    # Perform contraction using tensordot
    result_data = np.tensordot(tensor1.data, tensor2.data, axes=(axes1, axes2))
    
    # Determine result indices
    remaining_indices1 = [tensor1.indices[i] for i in range(tensor1.ndim) if i not in axes1]
    remaining_indices2 = [tensor2.indices[i] for i in range(tensor2.ndim) if i not in axes2]
    result_indices = tuple(remaining_indices1 + remaining_indices2)
    
    return Tensor(
        data=result_data,
        indices=result_indices,
        name=f"contract({tensor1.name or 'T1'},{tensor2.name or 'T2'})"
    )


def outer(tensor1: Tensor, tensor2: Tensor) -> Tensor:
    """
    Compute outer product of two tensors.
    
    The result has rank equal to the sum of input ranks.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Outer product tensor
        
    Examples:
        >>> v = Tensor(np.array([1, 2, 3]), indices=("^i",))
        >>> w = Tensor(np.array([4, 5]), indices=("^j",))
        >>> M = outer(v, w)
        >>> M.shape
        (3, 2)
    """
    result_data = np.outer(tensor1.data.ravel(), tensor2.data.ravel()).reshape(
        tensor1.shape + tensor2.shape
    )
    
    result_indices = tensor1.indices + tensor2.indices
    
    return Tensor(
        data=result_data,
        indices=result_indices,
        name=f"outer({tensor1.name or 'T1'},{tensor2.name or 'T2'})"
    )
