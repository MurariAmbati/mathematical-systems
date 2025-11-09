"""
Tensor flow visualization - visual graph of tensor operations and contractions.

Displays tensors as nodes and contractions as edges, showing:
- Tensor shapes and index labels
- Contraction patterns
- Value statistics (norm, max, min)
"""

from typing import List, Optional, Dict, Any
import numpy as np

from tas.core.tensor import Tensor


class TensorFlowGraph:
    """
    Graph representation of tensor operations and flows.
    
    Tracks tensors and their relationships through operations.
    """
    
    def __init__(self) -> None:
        """Initialize empty tensor flow graph."""
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
    
    def add_tensor(self, tensor: Tensor, node_id: Optional[str] = None) -> str:
        """
        Add a tensor as a node in the graph.
        
        Args:
            tensor: Tensor to add
            node_id: Optional explicit node ID
            
        Returns:
            Node ID
        """
        if node_id is None:
            node_id = f"tensor_{len(self.nodes)}"
        
        node = {
            "id": node_id,
            "name": tensor.name or node_id,
            "shape": tensor.shape,
            "indices": [str(idx) for idx in tensor.indices],
            "norm": float(tensor.norm()),
            "dtype": str(tensor.dtype)
        }
        
        self.nodes.append(node)
        return node_id
    
    def add_contraction(self, source_id: str, target_id: str, 
                       contracted_indices: List[str]) -> None:
        """
        Add a contraction edge between tensors.
        
        Args:
            source_id: Source tensor node ID
            target_id: Target tensor node ID
            contracted_indices: List of contracted index names
        """
        edge = {
            "source": source_id,
            "target": target_id,
            "type": "contraction",
            "indices": contracted_indices
        }
        
        self.edges.append(edge)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export graph to dictionary format.
        
        Returns:
            Dictionary with nodes and edges
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }


def visualize_tensor_flow(
    tensors: List[Tensor],
    operations: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> TensorFlowGraph:
    """
    Create a visual representation of tensor flow.
    
    Args:
        tensors: List of tensors to visualize
        operations: Optional list of operation descriptions
        output_path: Optional path to save visualization
        
    Returns:
        TensorFlowGraph object
        
    Note:
        Full implementation would use matplotlib or plotly to render.
        This is a simplified version that builds the graph structure.
    """
    graph = TensorFlowGraph()
    
    for i, tensor in enumerate(tensors):
        graph.add_tensor(tensor, node_id=f"T{i}")
    
    # In a full implementation, would analyze operations and add edges
    # For now, just return the graph structure
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(graph.to_dict(), f, indent=2)
    
    return graph


def plot_tensor_network(
    graph: TensorFlowGraph,
    layout: str = "spring",
    show: bool = True
) -> Any:
    """
    Plot tensor network diagram.
    
    Args:
        graph: TensorFlowGraph to plot
        layout: Layout algorithm ('spring', 'circular', 'hierarchical')
        show: Whether to display the plot
        
    Returns:
        Figure object (if matplotlib available)
        
    Note:
        Requires matplotlib. Full implementation would create
        an interactive diagram with node positions and edge routing.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError(
            "Tensor flow plotting requires matplotlib. "
            "Install with: pip install tensor-analysis-studio[viz]"
        )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Simple layout: arrange nodes in a circle
    n_nodes = len(graph.nodes)
    if n_nodes == 0:
        return fig
    
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 3.0
    
    positions = {
        node["id"]: (radius * np.cos(angle), radius * np.sin(angle))
        for node, angle in zip(graph.nodes, angles)
    }
    
    # Draw nodes
    for node in graph.nodes:
        x, y = positions[node["id"]]
        
        # Draw box for tensor
        box = patches.FancyBboxPatch(
            (x - 0.5, y - 0.3), 1.0, 0.6,
            boxstyle="round,pad=0.1",
            edgecolor="blue",
            facecolor="lightblue",
            linewidth=2
        )
        ax.add_patch(box)
        
        # Add label
        label = f"{node['name']}\n{node['shape']}"
        ax.text(x, y, label, ha="center", va="center", fontsize=10, weight="bold")
    
    # Draw edges
    for edge in graph.edges:
        source_pos = positions[edge["source"]]
        target_pos = positions[edge["target"]]
        
        ax.annotate(
            "", xy=target_pos, xytext=source_pos,
            arrowprops=dict(arrowstyle="->", lw=2, color="gray")
        )
    
    ax.set_xlim(-radius - 1, radius + 1)
    ax.set_ylim(-radius - 1, radius + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tensor Flow Network", fontsize=14, weight="bold")
    
    if show:
        plt.show()
    
    return fig
