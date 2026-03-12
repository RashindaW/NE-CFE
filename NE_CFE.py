#!/usr/bin/env python
"""NE-CFE: Node and Edge Counterfactual Explanations
====================================================
End‑to‑end pipeline for generating counterfactual explanations on graphs:
1. Train a GCN/ChebNet oracle on the full graph
2. Extract k‑hop factual sub‑graphs around test nodes
3. Jointly optimise node‑feature and edge perturbations to flip
   the oracle prediction with minimal change.

Supports multiple datasets (CiteSeer, Cora, Facebook, AIDS, Cornell).

Key design choices:
 • Correct discrete/continuous feature handling
 • Loss weighting: α trades node‑ vs edge‑sparsity, η gates the CE term
 • Perturbations *added* to base features, not overwritten
 • Undirected edge handling with self‑loops preserved
 • Per‑feature min/max clamping
 • Configurable α‑scheduling (linear, cosine, exponential, etc.)
 • Binary-Concrete (Gumbel-Softmax) relaxation for discrete features
 • Temperature annealing for the Concrete distribution
 • HEOM distance metric for heterogeneous feature spaces
"""

import os
import time
import argparse
from typing import Tuple, Dict, List
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch_geometric.datasets import Planetoid, TUDataset, WebKB
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    add_self_loops,
    k_hop_subgraph,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 0.   Parse arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="NE-CFE: Node and Edge Counterfactual Explanations")
parser.add_argument("--force-cpu", action="store_true", help="Force CPU use even if CUDA is available")
parser.add_argument("--cuda", type=int, default=0, help="CUDA device index to use (default: 0)")
parser.add_argument("--nodes", type=int, default=None, help="Number of test nodes to explain (None for all)")
parser.add_argument("--k-hop", type=int, default=2, help="Size of neighborhood subgraph (k-hop)")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for CF optimization")
parser.add_argument("--lr", type=float, default=0.002, help="Learning rate for CF optimization (recommended: 0.002)")
parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
parser.add_argument("--alpha-policy", type=str, default="linear", choices=["linear", "constant", "inverse", "cosine", "exponential", "sinusoidal", "dynamic"], 
                   help="Alpha scheduler policy (linear, constant, inverse, cosine, exponential, sinusoidal, dynamic)")
parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
parser.add_argument("--load-model", type=str, default=None, help="Path to load pretrained model instead of training")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--visualize", action="store_true", help="Enable visualization of the results")
# Temperature annealing arguments
parser.add_argument("--tau-start", type=float, default=1.0, help="Initial Concrete temperature τ₀")
parser.add_argument("--tau-min", type=float, default=0.1, help="Lowest temperature")
parser.add_argument("--tau-anneal", type=float, default=0.95, help="Multiplicative annealing per epoch (τ←τ·α)")
parser.add_argument("--heom-lambda", type=float, default=0.05, help="Weight for HEOM distance in the loss function")
parser.add_argument("--model-type", type=str, default="gcn", choices=["gcn", "chebnet"], 
                   help="Type of model to use (gcn or chebnet)")
# Dataset selection
parser.add_argument("--dataset", type=str, default="citeseer", choices=["citeseer", "cora", "facebook", "aids", "cornell"], 
                   help="Dataset to use (citeseer, cora, facebook, aids, cornell)")
# Dataset path argument (for custom datasets like Cornell)
parser.add_argument("--dataset-path", type=str, default=None, 
                   help="Path to custom dataset (required for cornell)")
# Split selection for WebKB (Cornell)
parser.add_argument("--split-idx", type=int, default=0, help="Split index for WebKB dataset (for Cornell)")
# Facebook and AIDS specific arguments
parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of training data (for Facebook/AIDS/Cornell)")
parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of validation data (for Facebook/AIDS/Cornell)")
args = parser.parse_args()

# Create results directory
os.makedirs(args.save_dir, exist_ok=True)

# Start timing the execution
start_time = time.time()

# -----------------------------------------------------------------------------
# 1.   Reproducibility & device
# -----------------------------------------------------------------------------
SEED = args.seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
if args.force_cpu:
    DEVICE = torch.device("cpu")
    print("Forcing CPU use as requested")
else:
    DEVICE = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# 2.   Dataset utilities
# -----------------------------------------------------------------------------
print(f"Loading {args.dataset.capitalize()} dataset...")
try:
    # Load the selected dataset
    if args.dataset in ["citeseer", "cora"]:
        # For Planetoid datasets (Citeseer and Cora)
        DATASET = Planetoid(root="./data", name=args.dataset.capitalize())
        FULL_GRAPH = DATASET[0].to(DEVICE)
        FULL_GRAPH.edge_index = to_undirected(FULL_GRAPH.edge_index)
    elif args.dataset == "cornell":
        # For Cornell dataset using WebKB
        try:
            # First try to use PyTorch Geometric's WebKB
            DATASET = WebKB(root="./data", name="Cornell")
            # WebKB datasets have a predefined split, use it
            data = DATASET[0]
            
            # Use the selected split or default to 0
            split_idx = min(args.split_idx, 9)  # Ensure in range 0-9
            
            # Create train/val/test masks using the provided splits
            train_mask = data.train_mask[:, split_idx]
            val_mask = data.val_mask[:, split_idx]
            test_mask = data.test_mask[:, split_idx]
            
            # Set masks
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            
            # Ensure the edge_index is undirected
            data.edge_index = to_undirected(data.edge_index)
            
            # Move to device
            FULL_GRAPH = data.to(DEVICE)
            
            print(f"Cornell dataset loaded via WebKB: {data.num_nodes} nodes, {data.num_edges} edges")
            print(f"Using split {split_idx}: Train {train_mask.sum().item()}, Val {val_mask.sum().item()}, Test {test_mask.sum().item()} nodes")
            
        except (ImportError, FileNotFoundError):
            # Fallback: Try to directly load the processed data.pt file if WebKB is not available
            if not args.dataset_path:
                raise ValueError("Cornell dataset could not be loaded via WebKB. Please provide --dataset-path.")
            cornell_path = args.dataset_path
            data_path = os.path.join(cornell_path, "processed/data.pt")
            
            if os.path.exists(data_path):
                # Load the PyTorch data file
                data = torch.load(data_path, map_location=DEVICE, weights_only=False)
                
                # Handle different data formats
                if isinstance(data, list):
                    data = data[0]  # Take the first graph if it's a list
                
                # Ensure we have the required attributes
                if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
                    # Create train/val/test masks if they don't exist
                    num_nodes = data.num_nodes
                    indices = torch.randperm(num_nodes)
                    
                    num_train = int(num_nodes * args.train_ratio)
                    num_val = int(num_nodes * args.val_ratio)
                    
                    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    
                    train_mask[indices[:num_train]] = True
                    val_mask[indices[num_train:num_train+num_val]] = True
                    test_mask[indices[num_train+num_val:]] = True
                    
                    data.train_mask = train_mask
                    data.val_mask = val_mask
                    data.test_mask = test_mask
                
                # Ensure the edge_index is undirected
                data.edge_index = to_undirected(data.edge_index)
                
                # Set the full graph
                FULL_GRAPH = data
                print(f"Cornell dataset loaded directly from data.pt: {data.num_nodes} nodes, {data.num_edges} edges")
            else:
                # Fallback to loading raw files - try to read the raw files
                raw_edges_path = os.path.join(cornell_path, "raw/out1_graph_edges.txt")
                raw_features_path = os.path.join(cornell_path, "raw/out1_node_feature_label.txt")
                
                if os.path.exists(raw_edges_path) and os.path.exists(raw_features_path):
                    # Read edge list
                    edges = []
                    with open(raw_edges_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                src, dst = int(parts[0]), int(parts[1])
                                edges.append([src, dst])
                    
                    # Read node features and labels
                    node_data = []
                    with open(raw_features_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:  # At least ID and label
                                node_id = int(parts[0])
                                label = int(parts[-1])  # Last column is the label
                                features = [float(x) for x in parts[1:-1]]  # All except ID and label
                                node_data.append((node_id, features, label))
                    
                    # Sort nodes by ID to ensure correct ordering
                    node_data.sort(key=lambda x: x[0])
                    
                    # Extract features and labels
                    features = [item[1] for item in node_data]
                    labels = [item[2] for item in node_data]
                    
                    # Convert to tensors
                    x = torch.tensor(features, dtype=torch.float).to(DEVICE)
                    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
                    edge_index = torch.tensor(edges, dtype=torch.long).t().to(DEVICE)
                    
                    # Create the data object
                    data = Data(x=x, edge_index=edge_index, y=y)
                    
                    # Create train/val/test masks
                    num_nodes = data.num_nodes
                    indices = torch.randperm(num_nodes)
                    
                    num_train = int(num_nodes * args.train_ratio)
                    num_val = int(num_nodes * args.val_ratio)
                    
                    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
                    
                    train_mask[indices[:num_train]] = True
                    val_mask[indices[num_train:num_train+num_val]] = True
                    test_mask[indices[num_train+num_val:]] = True
                    
                    data.train_mask = train_mask
                    data.val_mask = val_mask
                    data.test_mask = test_mask
                    
                    # Ensure the edge_index is undirected
                    data.edge_index = to_undirected(data.edge_index)
                    
                    # Set the full graph
                    FULL_GRAPH = data
                    print(f"Cornell dataset loaded from raw files: {data.num_nodes} nodes, {data.num_edges} edges")
                else:
                    raise FileNotFoundError(f"Could not find Cornell dataset files. Looked in {cornell_path}")
    elif args.dataset == "facebook":
        # For Facebook dataset
        from torch_geometric.datasets import FacebookPagePage
        
        DATASET = FacebookPagePage(root="./data")
        data = DATASET[0]
        
        # Create train/val/test masks if they don't exist
        if not hasattr(data, 'train_mask'):
            # First, determine the number of nodes for each split
            num_nodes = data.num_nodes
            num_train = int(num_nodes * args.train_ratio)
            num_val = int(num_nodes * args.val_ratio)
            num_test = num_nodes - num_train - num_val
            
            # Create a random permutation of node indices
            indices = torch.randperm(num_nodes)
            
            # Create masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            # Assign indices to each mask
            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train+num_val]] = True
            test_mask[indices[num_train+num_val:]] = True
            
            # Add masks to the data object
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        
        # Ensure the edge_index is undirected
        data.edge_index = to_undirected(data.edge_index)
        
        # Move to device
        FULL_GRAPH = data.to(DEVICE)
        
        # If there's no target for classification, we'll use page types as targets
        if not hasattr(FULL_GRAPH, 'y'):
            if hasattr(FULL_GRAPH, 'page_type'):
                # Use page types as targets
                FULL_GRAPH.y = FULL_GRAPH.page_type
            else:
                # If no suitable target exists, create synthetic communities using a clustering algorithm
                print("No target labels found. Creating synthetic communities using clustering...")
                from sklearn.cluster import KMeans
                
                # Extract node features
                features = FULL_GRAPH.x.cpu().numpy()
                
                # Apply KMeans clustering
                n_clusters = 4  # Can be adjusted
                kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
                clusters = kmeans.fit_predict(features)
                
                # Assign clusters as labels
                FULL_GRAPH.y = torch.tensor(clusters, dtype=torch.long).to(DEVICE)
                print(f"Created {n_clusters} synthetic communities as target labels")
    elif args.dataset == "aids":
        # For AIDS dataset from TUDataset
        DATASET = TUDataset(root="./data", name="AIDS")
        
        # Create a single large graph from all dataset graphs for our purposes
        # First, get all graph data
        all_graphs = [data for data in DATASET]
        
        # Sort by number of nodes (for stability)
        all_graphs.sort(key=lambda x: x.num_nodes)
        
        # Combine all graphs into one large graph with disconnected components
        total_nodes = sum(data.num_nodes for data in all_graphs)
        
        # Create new feature matrix
        combined_x = torch.zeros((total_nodes, DATASET.num_node_features), device=DEVICE)
        
        # Create new edge index
        edge_list = []
        
        # Create new labels for each node
        combined_y = torch.zeros(total_nodes, dtype=torch.long, device=DEVICE)
        
        # Track node offset as we add each graph
        node_offset = 0
        
        # Combine all graphs
        for i, data in enumerate(all_graphs):
            # Number of nodes in this graph
            num_nodes = data.num_nodes
            
            # Add features - ensure they're on the right device
            combined_x[node_offset:node_offset+num_nodes] = data.x.to(DEVICE)
            
            # Add edges (with node offset) - ensure they're on the right device
            if data.edge_index.size(1) > 0:  # Only if graph has edges
                offset_edges = data.edge_index.clone().to(DEVICE)
                offset_edges = offset_edges + node_offset
                edge_list.append(offset_edges)
            
            # Use the graph label as the label for each node
            graph_label = data.y.item()
            combined_y[node_offset:node_offset+num_nodes] = graph_label
            
            # Update offset
            node_offset += num_nodes
        
        # Combine edge indices
        if edge_list:
            combined_edge_index = torch.cat(edge_list, dim=1)
        else:
            combined_edge_index = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
        
        # Create the combined graph - explicitly set device for all tensors
        FULL_GRAPH = Data(
            x=combined_x,
            edge_index=combined_edge_index,
            y=combined_y
        ).to(DEVICE)  # Ensure the whole graph is on the right device
        
        # Ensure the edge_index is undirected
        FULL_GRAPH.edge_index = to_undirected(FULL_GRAPH.edge_index)
        
        # Create train/val/test masks
        num_nodes = FULL_GRAPH.num_nodes
        indices = torch.randperm(num_nodes)
        
        num_train = int(num_nodes * args.train_ratio)
        num_val = int(num_nodes * args.val_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        
        train_mask[indices[:num_train]] = True
        val_mask[indices[num_train:num_train+num_val]] = True
        test_mask[indices[num_train+num_val:]] = True
        
        FULL_GRAPH.train_mask = train_mask
        FULL_GRAPH.val_mask = val_mask
        FULL_GRAPH.test_mask = test_mask
        
        print(f"Created combined graph from {len(all_graphs)} AIDS graphs")
        print(f"Total nodes: {FULL_GRAPH.num_nodes}, total edges: {FULL_GRAPH.num_edges//2}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure PyTorch Geometric is properly installed.")
    exit(1)

N_FEATURES = FULL_GRAPH.num_node_features
N_CLASSES = len(FULL_GRAPH.y.unique())

# Analyze feature characteristics in detail
print(f"\nAnalyzing {args.dataset.capitalize()} features...")
feature_stats = {
    "min": FULL_GRAPH.x.min(dim=0)[0],
    "max": FULL_GRAPH.x.max(dim=0)[0],
    "mean": FULL_GRAPH.x.mean(dim=0),
    "std": FULL_GRAPH.x.std(dim=0),
    "sum": FULL_GRAPH.x.sum(dim=0),
    "nonzero": (FULL_GRAPH.x > 0).sum(dim=0),
}

# Check if features are actually binary
binary_check = ((FULL_GRAPH.x == 0) | (FULL_GRAPH.x == 1)).all()
print(f"All features are binary (0/1): {binary_check}")

# Count how many features have only 0/1 values
_is_binary = (FULL_GRAPH.x == 0) | (FULL_GRAPH.x == 1)
DISCRETE_MASK = _is_binary.all(dim=0)
CONTINUOUS_MASK = ~DISCRETE_MASK

# Feature bounds 
if args.dataset in ["citeseer", "cora", "cornell"]:
    # For citation networks, features should be binary (0/1)
    MIN_RANGE = torch.zeros(N_FEATURES, device=DEVICE)
    MAX_RANGE = torch.ones(N_FEATURES, device=DEVICE)
    
    # Override for any continuous features (if they exist)
    if CONTINUOUS_MASK.any():
        print("Setting custom min/max for continuous features")
        MIN_RANGE[CONTINUOUS_MASK] = feature_stats["min"][CONTINUOUS_MASK]
        MAX_RANGE[CONTINUOUS_MASK] = feature_stats["max"][CONTINUOUS_MASK]
else:
    # For Facebook and AIDS, use the actual min/max ranges
    MIN_RANGE = feature_stats["min"].clone()
    MAX_RANGE = feature_stats["max"].clone()
    
    # Override for discrete features (should be 0/1)
    if DISCRETE_MASK.any():
        MIN_RANGE[DISCRETE_MASK] = 0
        MAX_RANGE[DISCRETE_MASK] = 1

print(
    f"Nodes: {FULL_GRAPH.num_nodes} | Edges: {FULL_GRAPH.num_edges} | "
    f"Features: {N_FEATURES} (discrete: {DISCRETE_MASK.sum().item()}, continuous: {CONTINUOUS_MASK.sum().item()})"
)
print(f"Classes: {N_CLASSES} | Train: {FULL_GRAPH.train_mask.sum().item()} | " 
      f"Val: {FULL_GRAPH.val_mask.sum().item()} | Test: {FULL_GRAPH.test_mask.sum().item()}")

# Print the number of test nodes more clearly
test_set_size = FULL_GRAPH.test_mask.sum().item()
print(f"\nTEST SET SIZE: {test_set_size} nodes")
print(f"By default, the script will explain {min(args.nodes, test_set_size) if args.nodes is not None else test_set_size} nodes")

# -----------------------------------------------------------------------------
# 2.1.  New helper functions for Gumbel-softmax and HEOM distance
# -----------------------------------------------------------------------------

def tau_at(epoch: int, tau0: float = 1.0, tau_min: float = 0.05):
    """Calculate temperature at a given epoch with exponential decay"""
    return max(tau0 * (0.95 ** epoch), tau_min)

def sample_binary_concrete(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    """
    Sample from the Binary Concrete (Gumbel-Softmax) distribution
    
    Args:
        logits: unconstrained real numbers (same shape as binary feature-vector)
        tau: temperature
        hard: if True returns exact {0,1} via straight-through estimator
    
    Returns:
        Relaxed sample y ~ BinaryConcrete(σ(logits)) ∈ (0,1)
    """
    # Gumbel(0,1)
    eps = 1e-20
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + eps) + eps)
    y = torch.sigmoid((logits + g) / tau)  # (0,1)

    if hard:
        # Straight-through estimator
        y_hard = (y > 0.5).float()
        y = y_hard.detach() - y.detach() + y
        
    return y

def heom_distance(a: torch.Tensor, b: torch.Tensor,
                 disc_mask: torch.Tensor,
                 ranges: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Heterogeneous Euclidean-Overlap Metric
    
    Args:
        a, b: 1-D tensors (same device)
        disc_mask: bool mask for discrete dims
        ranges: (min, max) tensors for normalization
    
    Returns:
        scalar distance (no sqrt to keep it L2-like)
    """
    min_r, max_r = ranges
    
    # Continuous part (normalized)
    cont = (~disc_mask)
    denom = (max_r - min_r).clamp_min(1e-6)
    diff_cont = ((a[cont] - b[cont]) / denom[cont]) ** 2
    
    # Discrete part (overlap)
    diff_disc = (a[disc_mask] != b[disc_mask]).float()
    
    return torch.sum(diff_cont) + torch.sum(diff_disc)

# -----------------------------------------------------------------------------
# 3.   Oracle GCN
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Utils for Evaluation Metrics
# -----------------------------------------------------------------------------
def fidelity(oracle, node_idx, cf_graph, full_graph, y_true, target_local_idx, target_class):
    oracle.eval()
    with torch.no_grad():
        # Original prediction on full graph
        orig_logits = oracle(full_graph.x, full_graph.edge_index)
        y_hat = orig_logits[node_idx].argmax().item()
        
        # First term: whether oracle is correct on original input
        chi = int(y_hat == y_true)
        
        # Evaluate oracle directly on the counterfactual graph
        cf_logits = oracle(cf_graph.x, cf_graph.edge_index)
        y_hat_cf = cf_logits[target_local_idx].argmax().item()
        
        # Second term: whether CF still predicts ground truth
        indicator = int(y_hat_cf == y_true)
        
        # Paper's fidelity formula
        psi = chi - indicator
        
        # Fidelity calculation
        prediction_fidelity = 1 if y_hat == y_true else 0
        counterfactual_fidelity = 1 if y_hat_cf == y_true else 0
        fidelity_original = prediction_fidelity - counterfactual_fidelity
        
        # Simple validity: 1 iff we managed to build *any* CF AND it predicts the target class
        is_valid_simple = int(cf_graph is not None and y_hat_cf == target_class)
        
        return {
            "y_true": y_true,
            "orig_pred": y_hat,
            "cf_pred": y_hat_cf,
            "chi": chi,
            "psi": psi,  # Can be +1, 0, or -1
            "flipped": y_hat != y_hat_cf,  # Model-faithful measure
            "fidelity_original": fidelity_original,
            "valid_simple": is_valid_simple  # Simple validity (just needs to be the target class)
        }

def graph_embedding_distance(orig_graph, cf_graph, oracle):
    """Calculate the L2 distance between graph embeddings"""
    oracle.eval()
    with torch.no_grad():
        # Get hidden activations from first GCNConv
        h_orig = oracle.conv1(orig_graph.x, orig_graph.edge_index)
        h_cf = oracle.conv1(cf_graph.x, cf_graph.edge_index)
        
        # Mean-pool to get graph embeddings
        g_orig = h_orig.mean(dim=0)
        g_cf = h_cf.mean(dim=0)
        
        # Return L2 distance
        return torch.norm(g_orig - g_cf, p=2).item()

def calculate_dataset_mean_embedding(full_graph, oracle):
    """Calculate the mean embedding of the entire dataset"""
    oracle.eval()
    with torch.no_grad():
        # Get hidden activations from first GCNConv for the full graph
        h_full = oracle.conv1(full_graph.x, full_graph.edge_index)
        
        # Mean-pool to get the dataset's mean embedding
        mean_embedding = h_full.mean(dim=0)
        
        return mean_embedding

def calculate_distribution_distance(graph, oracle, dataset_mean_embedding):
    """Calculate the L2 distance between a graph's embedding and the dataset's mean embedding"""
    oracle.eval()
    with torch.no_grad():
        # Get hidden activations from first GCNConv
        h_graph = oracle.conv1(graph.x, graph.edge_index)
        
        # Mean-pool to get graph embedding
        graph_embedding = h_graph.mean(dim=0)
        
        # Return L2 distance to the dataset mean
        return torch.norm(graph_embedding - dataset_mean_embedding, p=2).item()

def calculate_feature_distance(orig_features, cf_features):
    """Calculate various distance metrics between original and CF features."""
    if orig_features is None or cf_features is None:
        return None
    
    # Convert to torch tensors if they're numpy arrays
    if isinstance(orig_features, np.ndarray):
        orig_features = torch.tensor(orig_features, device=DEVICE)
        cf_features = torch.tensor(cf_features, device=DEVICE)
    
    # L1 distance (Manhattan)
    l1_dist = torch.abs(orig_features - cf_features).sum().item()
    
    # L2 distance (Euclidean)
    l2_dist = torch.sqrt(torch.sum((orig_features - cf_features)**2)).item()
    
    # Cosine distance (1 - cosine similarity)
    cos_sim = F.cosine_similarity(orig_features.unsqueeze(0), 
                                 cf_features.unsqueeze(0), dim=1).item()
    cos_dist = 1.0 - cos_sim  # Convert similarity to distance
    
    # Hamming distance (for binary features, count of differing positions)
    hamming = (orig_features != cf_features).sum().item()
    
    # Add HEOM distance for mixed continuous/discrete features
    heom_dist = heom_distance(
        orig_features, 
        cf_features, 
        DISCRETE_MASK.to(orig_features.device),
        (MIN_RANGE.to(orig_features.device), MAX_RANGE.to(orig_features.device))
    ).item()
    
    return {
        "l1_distance": l1_dist,
        "l2_distance": l2_dist,
        "cosine_distance": cos_dist,
        "hamming_distance": hamming,
        "heom_distance": heom_dist
    }

def calculate_sparsity_metrics(res):
    """Calculate detailed sparsity metrics from a result."""
    metrics = {}
    
    # Node sparsity
    if res["feature_changes"] is not None:
        metrics["node_sparsity"] = {
            "absolute": res["feature_changes"],
            "relative": res["feature_changes"] / N_FEATURES if N_FEATURES > 0 else 0
        }
    
    # Edge sparsity - defined as the fraction of edges touching the target node that were removed
    if res["edge_changes"] is not None and res["total_edges"] is not None:
        metrics["edge_sparsity"] = {
            "absolute": res["edge_changes"],
            "relative": res["edge_changes"] / res["total_edges"] if res["total_edges"] > 0 else 0
        }
    
    return metrics

class GCN(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(n_in, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_out)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class ChebNet(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float = 0.5, K: int = 3):
        super().__init__()
        self.conv1 = ChebConv(n_in, n_hidden, K=K)
        self.conv2 = ChebConv(n_hidden, n_out, K=K)
        self.dropout = dropout
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


def train_gcn(model: nn.Module, data: Data, *, lr: float = 1e-2, weight_decay: float = 5e-4, epochs: int = 500) -> float:
    print(f"Training GCN with {epochs} epochs, lr={lr}, weight_decay={weight_decay}")
    print(f"Model architecture: {model}")
    
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = F.nll_loss
    model.train()
    
    # Training loop
    train_losses = []
    for epoch in tqdm(range(epochs), desc="Training GCN"):
        optim.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        clip_grad_norm_(model.parameters(), 2.0)
        optim.step()
        train_losses.append(loss.item())
        
        # Print training progress every 25 epochs
        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_pred = val_out.argmax(dim=1)
                val_acc = (val_pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs}: Train loss={loss.item():.4f}, Val acc={val_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_pred = logits.argmax(dim=1)
        test_acc = (test_pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        
        # Per-class accuracy
        class_correct = torch.zeros(N_CLASSES, device=DEVICE)
        class_total = torch.zeros(N_CLASSES, device=DEVICE)
        
        for i in range(len(test_pred)):
            if data.test_mask[i]:
                label = data.y[i].item()
                class_total[label] += 1
                if test_pred[i] == label:
                    class_correct[label] += 1
        
        class_acc = class_correct / class_total
        for c in range(N_CLASSES):
            if class_total[c] > 0:
                print(f"Class {c} accuracy: {class_acc[c].item():.4f} ({int(class_correct[c].item())}/{int(class_total[c].item())})")
    
    print(f"Oracle test accuracy: {test_acc:.4f}")
    return test_acc


print("\n" + "="*80)
print("TRAINING ORACLE GCN MODEL")
print("="*80)

if args.load_model and os.path.exists(args.load_model):
    print(f"Loading pretrained model from {args.load_model}")
    if args.model_type == "gcn":
        ORACLE = GCN(N_FEATURES, 16, N_CLASSES).to(DEVICE)
    else:  # chebnet
        ORACLE = ChebNet(N_FEATURES, 16, N_CLASSES, K=3).to(DEVICE)
    ORACLE.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    
    # Quick validation
    ORACLE.eval()
    with torch.no_grad():
        logits = ORACLE(FULL_GRAPH.x, FULL_GRAPH.edge_index)
        test_pred = logits.argmax(dim=1)
        test_acc = (test_pred[FULL_GRAPH.test_mask] == FULL_GRAPH.y[FULL_GRAPH.test_mask]).float().mean().item()
    print(f"Loaded Oracle test accuracy: {test_acc:.4f}")
else:
    if args.model_type == "gcn":
        ORACLE = GCN(N_FEATURES, 16, N_CLASSES).to(DEVICE)
        print(f"Using GCN model")
    else:  # chebnet
        ORACLE = ChebNet(N_FEATURES, 16, N_CLASSES, K=3).to(DEVICE)
        print(f"Using ChebNet model with K={ORACLE.K}")
    
    oracle_acc = train_gcn(ORACLE, FULL_GRAPH, epochs=500)
    
    # Save the Oracle model
    model_path = os.path.join(args.save_dir, f"oracle_{args.dataset}_{args.model_type}_model.pt")
    torch.save(ORACLE.state_dict(), model_path)
    print(f"Oracle model saved to {model_path}")

# Freeze oracle weights
for p in ORACLE.parameters():
    p.requires_grad_(False)

# -----------------------------------------------------------------------------
# 4.   GraphPerturber
# -----------------------------------------------------------------------------

class GraphPerturber(nn.Module):
    """Optimises node‑feature (P_x) and edge‑weight (EP_x) perturbations for **one**
    target node inside a *factual* sub‑graph.
    """

    def __init__(
        self,
        factual: Data,
        oracle: nn.Module,
        node_idx: int,
    ) -> None:
        super().__init__()
        self.oracle = oracle
        self.device = factual.x.device
        self.node_idx = node_idx

        # Build mask for edges touching the target node inside the factual graph
        src, dst = factual.edge_index
        self.register_buffer(
            "edge_touch_mask",
            (src == node_idx) | (dst == node_idx),
        )
        
        # Count edges that touch the target node
        num_touching_edges = int(self.edge_touch_mask.sum().item())
        print(f"  Target node has {num_touching_edges} connections")

        # parameters ----------------------------------------------------------
        self.P_x = Parameter(torch.zeros(factual.num_node_features, device=self.device))
        # Only create parameters for edges that touch the target node
        self.EP_x = Parameter(torch.zeros(num_touching_edges, device=self.device))

        # buffers --------------------------------------------------------------
        self.register_buffer("orig_x", factual.x.clone(), persistent=False)
        self.register_buffer("edge_index", factual.edge_index.clone(), persistent=False)

        # masks & ranges -------------------------------------------------------
        self.register_buffer("disc_mask", DISCRETE_MASK)
        self.register_buffer("cont_mask", CONTINUOUS_MASK)
        self.register_buffer("min_r", MIN_RANGE)
        self.register_buffer("max_r", MAX_RANGE)
        
        # Track current epoch for temperature annealing
        self._epoch = 0

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _hard_round(t: torch.Tensor) -> torch.Tensor:
        return t.round().clamp(0.0, 1.0)

    # ---------------------------------------------------------------------
    # forward passes
    # ---------------------------------------------------------------------

    def _apply_perturbations(self, hard: bool = False, tau: float = 0.5):
        """Return (x', ei', ew') without in‑place ops that break autograd."""
        # ---- node features -------------------------------------------------
        x_new = self.orig_x.clone()                     # safe base tensor

        # Binary features with Gumbel-softmax relaxation
        disc_logits = self.P_x * self.disc_mask         # mask keeps grads
        disc_probs = sample_binary_concrete(disc_logits, tau, hard) * self.disc_mask

        # Continuous features without tanh (direct application)
        cont_delta = self.P_x * self.cont_mask
        cont_part = (x_new[self.node_idx] + cont_delta) * self.cont_mask
        
        # Combine and clamp to valid range
        new_feat = torch.clamp(disc_probs + cont_part, self.min_r, self.max_r)

        x_new[self.node_idx] = new_feat                # single assignment

        # ---- edge weights --------------------------------------------------
        ew = torch.ones(self.edge_index.size(1), device=x_new.device)
        ew[self.edge_touch_mask] = torch.sigmoid(self.EP_x)
        if hard:
            ew[self.edge_touch_mask] = self._hard_round(ew[self.edge_touch_mask])

        ei, ew = add_self_loops(self.edge_index, ew)
        return x_new, ei, ew

    def forward(self, tau: float = None):
        """Pass the perturbed graph through the oracle."""
        # Use provided tau or default
        if tau is None:
            tau = tau_at(self._epoch, args.tau_start, args.tau_min)
        self._epoch += 1
        
        x_p, ei, ew = self._apply_perturbations(hard=False, tau=tau)
        return self.oracle(x_p, ei, ew)

    # ---------------------------------------------------------------------
    # losses
    # ---------------------------------------------------------------------

    def node_loss(self):
        """Compute sparsity loss for node features"""
        # For discrete features: binary cross-entropy loss
        disc_probs = torch.sigmoid(self.P_x)
        # Target is the original feature value for discrete features
        # Use BCE loss which is appropriate for binary features
        disc_loss = F.binary_cross_entropy(
            disc_probs[self.disc_mask], 
            self.orig_x[self.node_idx][self.disc_mask]
        ) if self.disc_mask.any() else 0.0
        
        # Increase weight for discrete features to counter entropy from Gumbel-softmax
        disc_loss = disc_loss * 2.0
        
        # For continuous features (if any): MSE loss to original
        cont_delta = self.P_x  # Removed tanh
        cont_new = self.orig_x[self.node_idx] + cont_delta * self.cont_mask
        cont_loss = F.mse_loss(
            cont_new[self.cont_mask],
            self.orig_x[self.node_idx][self.cont_mask]
        ) if self.cont_mask.any() else 0.0
        
        return disc_loss + cont_loss

    def edge_loss(self):
        edge_w = torch.sigmoid(self.EP_x)
        return torch.abs(edge_w - 1.0).sum()
        
    def heom_loss(self):
        """Compute HEOM distance between original and perturbed features"""
        with torch.no_grad():
            x_p, _, _ = self._apply_perturbations(hard=True)
            orig_feat = self.orig_x[self.node_idx]
            pert_feat = x_p[self.node_idx]
            
        return heom_distance(orig_feat, pert_feat, self.disc_mask, (self.min_r, self.max_r))

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------

    def discretised_prediction(self, tau: float = 0.1):
        with torch.no_grad():
            x_p, ei, ew = self._apply_perturbations(hard=True, tau=tau)
            return self.oracle(x_p, ei, ew)

    def build_cf_graph(self, tau: float = 0.1):
        x_p, ei, ew = self._apply_perturbations(hard=True, tau=tau)
        keep_mask = ew > 0.5
        return Data(
            x=x_p, 
            edge_index=ei[:, keep_mask], 
            orig_x=self.orig_x,  # Store original features
            edge_index_orig=self.edge_index  # Store original edges
        )

    def get_feature_changes(self, tau: float = 0.1):
        with torch.no_grad():
            x_p, ei, ew = self._apply_perturbations(hard=True, tau=tau)

            # --- node‑feature diff --------------------------------------------
            orig_feat = self.orig_x[self.node_idx]
            pert_feat = x_p[self.node_idx]
            feat_changes = (orig_feat != pert_feat).sum().item()

            # --- edge diff (ignore the self‑loops we just added) --------------
            n_orig_edges = self.edge_index.size(1)          # before add_self_loops
            keep_orig    = ew[:n_orig_edges] > 0.5          # slice back to E
            removed_edges = (self.edge_touch_mask & ~keep_orig).sum().item()
            total_touching = self.edge_touch_mask.sum().item()

            return feat_changes, orig_feat, pert_feat, removed_edges, total_touching


# -----------------------------------------------------------------------------
# 5.   Counterfactual generation for a single node
# -----------------------------------------------------------------------------

def node_sparsity_original(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the node sparsity between the factual and counterfactual graphs.
    This follows the original implementation.
    """
    # Calculate the number of modified node attributes
    modified_attributes = torch.sum(factual.x != counterfactual.x)
    # Calculate the node sparsity
    sparsity = modified_attributes / factual.x.numel()
    return sparsity.item()

def edge_sparsity_original(factual: Data, counterfactual: Data) -> float:
    """
    Calculate the edge sparsity between the factual and counterfactual graphs using edge indices.
    This follows the original implementation.
    """
    # Get the edge indices
    factual_edges = set(map(tuple, factual.edge_index.t().tolist()))
    counterfactual_edges = set(map(tuple, counterfactual.edge_index.t().tolist()))
    
    # Calculate the number of modified edges
    modified_edges = len(factual_edges.symmetric_difference(counterfactual_edges))
    
    # Calculate the total number of edges in the factual graph
    total_edges = len(factual_edges)
    
    # Calculate the edge sparsity
    sparsity = modified_edges / total_edges if total_edges > 0 else 0
    return sparsity

def generate_counterfactual(
    full_graph: Data,
    oracle: nn.Module,
    node_idx: int,
    *,
    k_hop: int = 2,
    epochs: int = 150,
    lr: float = 0.001,
    alpha_policy: str = "linear",
    patience: int = 20,
    verbose: bool = False,
    debug: bool = False,
) -> Dict:
    # factual sub‑graph -------------------------------------------------------
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx,
        k_hop,
        full_graph.edge_index,
        relabel_nodes=True,
    )
    factual = Data(
        x=full_graph.x[sub_nodes],
        edge_index=sub_edge_index,
    ).to(full_graph.x.device)

    target_local_idx = int(mapping)  # node position inside sub‑graph
    
    if verbose:
        print(f"  Subgraph size: {len(sub_nodes)} nodes, {sub_edge_index.size(1)//2} edges")

    # original prediction ----------------------------------------------------
    with torch.no_grad():
        orig_logits = oracle(full_graph.x, full_graph.edge_index)
        orig_pred = int(orig_logits[node_idx].argmax())
        orig_prob = orig_logits[node_idx, orig_pred].exp().item()
        
        # Get probability for all classes
        probs = orig_logits[node_idx].exp()
        if verbose:
            print(f"  Original probabilities: ", end="")
            for c in range(N_CLASSES):
                print(f"Class {c}: {probs[c].item():.4f}  ", end="")
            print()

    # Get the 2nd highest probability class as target (next most likely)
    # First, create a copy of logits and set the highest value to negative infinity
    target_logits = orig_logits[node_idx].clone()
    target_logits[orig_pred] = float('-inf')
    target_class = int(target_logits.argmax())
    target_prob = probs[target_class].item()
    
    if verbose:
        print(f"  Original prediction: Class {orig_pred} ({orig_prob:.4f})")
        print(f"  Target class: Class {target_class} ({target_prob:.4f})")

    # perturber setup --------------------------------------------------------
    try:
        perturber = GraphPerturber(factual, oracle, node_idx=target_local_idx).to(DEVICE)
        opt = torch.optim.Adam(perturber.parameters(), lr=lr)
    except Exception as e:
        print(f"  Error creating perturber: {e}")
        return {
            "orig_pred": orig_pred,
            "orig_prob": orig_prob,
            "target_class": target_class,
            "cf_graph": None,
            "valid": False,
            "error": str(e)
        }

    # Alpha scheduler implementations
    def alpha_at(e: int, edge_loss: float = None, node_loss: float = None):
        if alpha_policy == "linear":
            return max(1.0 - e / epochs, 0.0)
        elif alpha_policy == "constant":
            return 0.5  # default const
        elif alpha_policy == "inverse":
            return 1.0 / (1.0 + e / (epochs / 10))
        elif alpha_policy == "cosine":
            return 0.5 * (1 + np.cos(np.pi * e / epochs))
        elif alpha_policy == "exponential":
            return max(0.0, np.exp(-e / (epochs / 10)))
        elif alpha_policy == "sinusoidal":
            return max(0.0, 0.5 * (1 + np.cos(np.pi * e / epochs)))
        elif alpha_policy == "dynamic":
            # Dynamic adjustment based on loss values - prioritize whichever needs more attention
            if edge_loss is not None and node_loss is not None:
                return 0.0 if edge_loss > node_loss else 1.0
            return 0.5  # Default if losses not provided
        return 0.5  # default

    # Initialize temperature for Gumbel-softmax
    tau = args.tau_start
    tau_min = args.tau_min
    tau_anneal = args.tau_anneal

    best_cf = None
    best_loss = float("inf")
    
    # Track metrics
    losses = {"total": [], "pred": [], "node": [], "edge": [], "heom": []}
    
    # Early stopping tracking
    no_improvement = 0
    
    # Progress bar
    if verbose:
        pbar = tqdm(range(epochs), desc=f"  Optimizing CF for node {node_idx}")
    else:
        pbar = range(epochs)

    for epoch in pbar:
        opt.zero_grad()
        
        # Forward pass with current temperature
        logits = perturber(tau)
        
        # Use proper nll_loss instead of raw negative log prob to avoid inf/-inf
        pred_loss = F.nll_loss(
            logits[[target_local_idx]],
            torch.tensor([target_class], device=DEVICE)
        )

        # Sparsity losses
        node_sparsity = perturber.node_loss()
        edge_sparsity = perturber.edge_loss()
        heom_loss = perturber.heom_loss()
        
        # Current alpha value
        alpha = alpha_at(epoch, edge_sparsity.item(), node_sparsity.item())

        # indicator η (turns off CE once flip achieved in **hard** graph)
        with torch.no_grad():
            hard_pred = perturber.discretised_prediction(tau=tau_min)[target_local_idx].argmax().item()
            eta = 1.0 if hard_pred != target_class else 0.0

        # Total loss with HEOM component
        total = eta * pred_loss + (1 - alpha) * edge_sparsity + alpha * node_sparsity + args.heom_lambda * heom_loss
        
        # Track losses
        losses["total"].append(total.item())
        losses["pred"].append(pred_loss.item())
        losses["node"].append(node_sparsity.item())
        losses["edge"].append(edge_sparsity.item())
        losses["heom"].append(heom_loss.item())
        
        # Logging in verbose mode
        if verbose and isinstance(pbar, tqdm) and epoch % 10 == 0:
            pbar.set_postfix({
                "loss": f"{total.item():.4f}", 
                "pred": f"{hard_pred}",
                "alpha": f"{alpha:.2f}",
                "eta": f"{eta:.0f}",
                "tau": f"{tau:.2f}"
            })
        
        # Compute gradients
        total.backward()
        
        # Debug output for gradients
        if debug and epoch % 20 == 0:
            with torch.no_grad():
                px_grad_norm = torch.norm(perturber.P_x.grad).item()
                ex_grad_norm = torch.norm(perturber.EP_x.grad).item()
                print(f"  [Debug] Epoch {epoch}: P_x grad={px_grad_norm:.4e}, EP_x grad={ex_grad_norm:.4e}")
        
        # Clip and apply gradients
        clip_grad_norm_(perturber.parameters(), 2.0)
        opt.step()
        
        # Anneal temperature
        tau = max(tau * tau_anneal, tau_min)

        # track best
        with torch.no_grad():
            if hard_pred == target_class and total.item() < best_loss:
                best_loss = total.item()
                best_cf = perturber.build_cf_graph(tau=tau_min)
                no_improvement = 0
                if verbose:
                    changes, _, _, edge_changes, total_edges = perturber.get_feature_changes(tau=tau_min)
                    print(f"  [Epoch {epoch}] Found valid CF: {changes} feature changes, loss={total.item():.4f}")
            else:
                no_improvement += 1
                
            # Early stopping
            if no_improvement >= patience and best_cf is not None:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Get feature and edge changes for the best counterfactual
    feature_changes = None
    orig_features = None
    cf_features = None
    edge_changes = None
    total_edges = None
    
    if best_cf is not None:
        feature_changes, orig_features, cf_features, edge_changes, total_edges = perturber.get_feature_changes(tau=tau_min)

    # Create factual graph for embedding distance calculation if we have a valid CF
    emb_distance = None
    fidelity_metrics = None
    
    if best_cf is not None:
        # Create factual graph for embedding distance calculation
        factual_graph = Data(
            x=best_cf.orig_x,
            edge_index=best_cf.edge_index_orig
        ).to(DEVICE)
        
        # Calculate embedding distance
        emb_distance = graph_embedding_distance(factual_graph, best_cf, ORACLE)
        
        # Use the full graph's ground truth label (y) for the node
        ground_truth = FULL_GRAPH.y[node_idx].item()
        fidelity_metrics = fidelity(ORACLE, node_idx, best_cf, FULL_GRAPH, ground_truth, target_local_idx, target_class)
    
    # Create the results dictionary first
    res = {
        "orig_pred": orig_pred,
        "orig_prob": orig_prob,
        "target_class": target_class,
        "cf_graph": best_cf,
        "valid": best_cf is not None and (fidelity_metrics["cf_pred"] == target_class if fidelity_metrics else False),
        "losses": losses,
        "feature_changes": feature_changes,
        "edge_changes": edge_changes,
        "total_edges": total_edges,
        "target_local_idx": target_local_idx,  # Keep for fidelity computation
        "orig_features": orig_features.cpu().numpy() if orig_features is not None else None,
        "cf_features": cf_features.cpu().numpy() if cf_features is not None else None
    }
    
    # Now calculate additional metrics using the res dictionary
    sparsity_metrics = calculate_sparsity_metrics(res)
    feature_metrics = calculate_feature_distance(orig_features, cf_features)
    
    # Add the additional metrics to the results
    res["sparsity_metrics"] = sparsity_metrics
    res["feature_metrics"] = feature_metrics
    res["embedding_distance"] = emb_distance
    res["fidelity_metrics"] = fidelity_metrics
    
    # Calculate original sparsity metrics if we have a valid counterfactual
    if best_cf is not None:
        # Create factual data with the structure expected by original metrics
        factual = Data(
            x=best_cf.orig_x,
            edge_index=best_cf.edge_index_orig
        ).to(DEVICE)
        
        # Calculate original sparsity metrics
        node_sparsity_orig = node_sparsity_original(factual, best_cf)
        edge_sparsity_orig = edge_sparsity_original(factual, best_cf)
        
        # Store original metrics
        res["original_metrics"] = {
            "node_sparsity": node_sparsity_orig,
            "edge_sparsity": edge_sparsity_orig
        }
    else:
        res["original_metrics"] = None
    
    # Calculate distribution distance if we have a valid counterfactual
    if best_cf is not None:
        distribution_distance = calculate_distribution_distance(best_cf, ORACLE, calculate_dataset_mean_embedding(FULL_GRAPH, ORACLE))
        res["distribution_distance"] = distribution_distance
    else:
        res["distribution_distance"] = None

    return res


# -----------------------------------------------------------------------------
# 6.   Run explanations on test nodes
# -----------------------------------------------------------------------------

print("\n" + "="*80)
print(f"GENERATING COUNTERFACTUAL EXPLANATIONS FOR {args.nodes} TEST NODES")
print("="*80)

# Find test nodes
test_nodes = torch.nonzero(FULL_GRAPH.test_mask).flatten().tolist()
if args.nodes is not None:
    test_nodes = test_nodes[:args.nodes]
    print(f"Using {len(test_nodes)} test nodes (limited by --nodes argument)")
else:
    print(f"Using all {len(test_nodes)} test nodes")
results: List[Dict] = []

# Run explanations
for i, n in enumerate(test_nodes):
    print(f"\n[{i+1}/{len(test_nodes)}] Explaining node {n}...")
    start_node = time.time()
    res = generate_counterfactual(
        FULL_GRAPH, 
        ORACLE, 
        n, 
        k_hop=args.k_hop,
        epochs=args.epochs,
        lr=args.lr,
        alpha_policy=args.alpha_policy,
        patience=args.patience,
        verbose=True,
        debug=args.debug
    )
    node_time = time.time() - start_node
    
    # Add timing information
    res["time"] = node_time
    results.append(res)
    
    # Print summary
    print(f"  Valid: {res['valid']} | From class {res['orig_pred']} → {res['target_class']}")
    if res["valid"]:
        # Calculate additional metrics
        sparsity_metrics = res["sparsity_metrics"]
        feature_metrics = res["feature_metrics"]
        
        # Use the full graph's ground truth label (y) for the node
        ground_truth = FULL_GRAPH.y[n].item()
        fidelity_metrics = res["fidelity_metrics"]
        
        # Add to results
        res["sparsity_metrics"] = sparsity_metrics
        res["feature_metrics"] = feature_metrics
        res["fidelity_metrics"] = fidelity_metrics
        
        # Print summary
        node_sparsity = sparsity_metrics.get("node_sparsity", {}).get("relative", 0) * 100
        edge_sparsity = sparsity_metrics.get("edge_sparsity", {}).get("relative", 0) * 100
        
        # Original metrics
        orig_node_sparsity = res["original_metrics"]["node_sparsity"] * 100
        orig_edge_sparsity = res["original_metrics"]["edge_sparsity"] * 100
        
        print(f"  Feature changes: {res['feature_changes']} ({node_sparsity:.1f}%) | "
              f"Edge changes: {res['edge_changes']}/{res['total_edges']} ({edge_sparsity:.1f}%) | "
              f"Orig-Node: {orig_node_sparsity:.1f}% | Orig-Edge: {orig_edge_sparsity:.1f}% | "
              f"Flipped: {fidelity_metrics['flipped']} | Simple Valid: {fidelity_metrics['valid_simple']} | "
              f"Psi: {fidelity_metrics['psi']} | Original Fidelity: {fidelity_metrics['fidelity_original']} | "
              f"HEOM dist: {res['feature_metrics']['heom_distance']:.4f} | "
              f"Embedding dist: {res['embedding_distance']:.4f} | "
              f"Distribution dist: {res['distribution_distance']:.4f} | "
              f"Time: {node_time:.2f}s")
    else:
        print(f"  No valid counterfactual found | Time: {node_time:.2f}s")

# Calculate summary stats
valids = sum(r["fidelity_metrics"]["flipped"] 
             for r in results 
             if r.get("fidelity_metrics"))
success_rate = valids / len(results) * 100

# Simple-validity is now defined "per-CF"; evaluate it only on the nodes where a CF was returned
valid_cf_results = [r for r in results if r.get("fidelity_metrics")]
simple_valids = sum(r["fidelity_metrics"]["valid_simple"] for r in valid_cf_results)
simple_success_rate = 0 if not valid_cf_results else simple_valids / len(valid_cf_results) * 100

avg_feat_changes = np.mean([r["feature_changes"] for r in results if r["valid"] and r["feature_changes"] is not None])
avg_edge_changes = np.mean([r["edge_changes"] for r in results if r["valid"] and r["edge_changes"] is not None])
avg_time = np.mean([r["time"] for r in results])

# Calculate average fidelity
flipped_results = [r["fidelity_metrics"]["flipped"] for r in results if r["valid"] and "fidelity_metrics" in r]
psi_results = [r["fidelity_metrics"]["psi"] for r in results if r["valid"] and "fidelity_metrics" in r]
fidelity_original_results = [r["fidelity_metrics"]["fidelity_original"] for r in results if r["valid"] and "fidelity_metrics" in r]
valid_simple_results = [r["fidelity_metrics"]["valid_simple"] for r in results if r["valid"] and "fidelity_metrics" in r]
avg_flipped = np.mean(flipped_results) if flipped_results else 0
avg_psi = np.mean(psi_results) if psi_results else 0
avg_fidelity_original = np.mean(fidelity_original_results) if fidelity_original_results else 0
avg_valid_simple = np.mean(valid_simple_results) if valid_simple_results else 0

# Calculate average distances
l1_distances = [r["feature_metrics"]["l1_distance"] for r in results if r["valid"] and "feature_metrics" in r]
l2_distances = [r["feature_metrics"]["l2_distance"] for r in results if r["valid"] and "feature_metrics" in r]
cosine_sims = [r["feature_metrics"]["cosine_distance"] for r in results if r["valid"] and "feature_metrics" in r]
hamming_dists = [r["feature_metrics"]["hamming_distance"] for r in results if r["valid"] and "feature_metrics" in r]
heom_dists = [r["feature_metrics"]["heom_distance"] for r in results if r["valid"] and "feature_metrics" in r]
embedding_dists = [r["embedding_distance"] for r in results if r["valid"] and "embedding_distance" in r]

avg_l1 = np.mean(l1_distances) if l1_distances else 0
avg_l2 = np.mean(l2_distances) if l2_distances else 0
avg_cosine = np.mean(cosine_sims) if cosine_sims else 0
avg_hamming = np.mean(hamming_dists) if hamming_dists else 0
avg_heom = np.mean(heom_dists) if heom_dists else 0
avg_embedding = np.mean(embedding_dists) if embedding_dists else 0

# Calculate average sparsity
node_sparsity_rel = [r["sparsity_metrics"]["node_sparsity"]["relative"] for r in results 
                    if r["valid"] and "sparsity_metrics" in r and "node_sparsity" in r["sparsity_metrics"]]
edge_sparsity_rel = [r["sparsity_metrics"]["edge_sparsity"]["relative"] for r in results 
                    if r["valid"] and "sparsity_metrics" in r and "edge_sparsity" in r["sparsity_metrics"]]

# Calculate original metrics
node_sparsity_orig = [r["original_metrics"]["node_sparsity"] for r in results 
                     if r["valid"] and "original_metrics" in r]
edge_sparsity_orig = [r["original_metrics"]["edge_sparsity"] for r in results 
                     if r["valid"] and "original_metrics" in r]

avg_node_sparsity = np.mean(node_sparsity_rel) if node_sparsity_rel else 0
avg_edge_sparsity = np.mean(edge_sparsity_rel) if edge_sparsity_rel else 0
avg_node_sparsity_orig = np.mean(node_sparsity_orig) if node_sparsity_orig else 0
avg_edge_sparsity_orig = np.mean(edge_sparsity_orig) if edge_sparsity_orig else 0

# Calculate oracle accuracy on explained nodes
chis = [r["fidelity_metrics"]["chi"]
        for r in results if r.get("fidelity_metrics")]
avg_chi = np.mean(chis) if chis else 0

# Calculate distribution distance
distribution_dists = [r["distribution_distance"] for r in results if r["valid"] and "distribution_distance" in r]
avg_distribution = np.mean(distribution_dists) if distribution_dists else 0

# Save results with structured directory path
results_path = os.path.join(
    args.save_dir,
    f"{args.dataset}/{args.model_type}/{args.alpha_policy}",
    f"cf_results_{datetime.now():%Y%m%d_%H%M%S}.pt"
)
os.makedirs(os.path.dirname(results_path), exist_ok=True)

torch.save({
    "results": results,
    "summary": {
        "n_test_nodes": len(test_nodes),
        "n_valid": valids,
        "success_rate": success_rate,
        "n_valid_simple": simple_valids,
        "simple_success_rate": simple_success_rate,
        "avg_feat_changes": avg_feat_changes,
        "avg_edge_changes": avg_edge_changes,
        "avg_node_sparsity": avg_node_sparsity,
        "avg_edge_sparsity": avg_edge_sparsity,
        "avg_orig_node_sparsity": avg_node_sparsity_orig,
        "avg_orig_edge_sparsity": avg_edge_sparsity_orig,
        "avg_flipped": avg_flipped,
        "avg_psi": avg_psi,
        "avg_fidelity_original": avg_fidelity_original,
        "avg_valid_simple": avg_valid_simple,
        "avg_chi": avg_chi,
        "avg_l1_distance": avg_l1,
        "avg_l2_distance": avg_l2,
        "avg_cosine_similarity": avg_cosine,
        "avg_hamming_distance": avg_hamming,
        "avg_heom_distance": avg_heom,
        "avg_embedding_distance": avg_embedding,
        "avg_distribution_distance": avg_distribution,
        "avg_time": avg_time,
        "total_time": time.time() - start_time,
    }
}, results_path)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Total nodes explained: {len(results)}")
print(f"Valid counterfactuals (prediction flipped): {valids}/{len(results)} ({success_rate:.1f}%)")
print(f"Simple validity (target class reached): {simple_valids}/{len(results)} ({simple_success_rate:.1f}%)")
if valids > 0:
    print(f"Oracle accuracy on explained nodes: {avg_chi*100:.1f}%")
    print(f"Average fidelity (Ψ): {avg_psi:.2f}")
    print(f"Average original fidelity: {avg_fidelity_original:.2f}")
    print(f"Average feature changes: {avg_feat_changes:.2f} ({avg_node_sparsity*100:.1f}%)")
    print(f"Average edge changes: {avg_edge_changes:.2f} ({avg_edge_sparsity*100:.1f}%)")
    print(f"Original metrics:")
    print(f"  Node sparsity: {avg_node_sparsity_orig*100:.1f}%")
    print(f"  Edge sparsity: {avg_edge_sparsity_orig*100:.1f}%")
    print(f"Distribution distances:")
    print(f"  L1 (Manhattan): {avg_l1:.2f}")
    print(f"  L2 (Euclidean): {avg_l2:.2f}")
    print(f"  Cosine distance: {avg_cosine:.4f}")
    print(f"  Hamming distance: {avg_hamming:.2f}")
    print(f"  HEOM distance: {avg_heom:.4f}")
    print(f"  Embedding distance: {avg_embedding:.4f}")
    print(f"  Distribution distance: {avg_distribution:.4f}")
print(f"Average time per node: {avg_time:.2f}s")
print(f"Total execution time: {time.time() - start_time:.2f}s")
print(f"Results saved to: {results_path}")
print("\nDone!")

# Optionally create a visualization for one example
if args.visualize:
    try:
        import matplotlib.pyplot as plt
        
        # Find a valid example
        valid_examples = [r for r in results if r["valid"]]
        if valid_examples:
            example = valid_examples[0]
            
            plt.figure(figsize=(12, 6))
            
            # Plot feature comparison
            plt.subplot(1, 2, 1)
            plt.bar(range(N_FEATURES), example["orig_features"], alpha=0.5, label="Original")
            plt.bar(range(N_FEATURES), example["cf_features"], alpha=0.5, label="Counterfactual")
            plt.xlabel("Feature Index")
            plt.ylabel("Feature Value")
            plt.title(f"Feature Comparison (Node {test_nodes[results.index(example)]})")
            plt.legend()
            
            # Plot changes
            plt.subplot(1, 2, 2)
            changes = np.abs(example["cf_features"] - example["orig_features"])
            plt.bar(range(N_FEATURES), changes)
            plt.xlabel("Feature Index")
            plt.ylabel("Change Magnitude")
            plt.title(f"Feature Changes (Total: {example['feature_changes']})")
            
            # Save figure
            fig_path = os.path.join(args.save_dir, f"{args.dataset}_feature_changes.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            print(f"Visualization saved to: {fig_path}")
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")
    except Exception as e:
        print(f"Error creating visualization: {e}")

# For Facebook-specific visualizations with networkx, add the necessary import
if args.dataset == "facebook" and args.visualize:
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX not installed. Some visualizations may be limited.") 
