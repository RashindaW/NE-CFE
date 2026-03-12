import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
import numpy as np
import sys
import time
import argparse
import os
from datetime import datetime
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.data import Data
import pandas as pd

# Import explainer classes from individual implementations
from ego_counterfactual_trials import EGOExplainer
from cff_counterfactual_trials import CFFExplainer
from cf_gnn_counterfactual_trials import CFGNNExplainer
from random_counterfactual_trials import RandomExplainer

# GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

# ChebNet model
class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def heom_distance(a, b, disc_mask, ranges):
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

def calculate_feature_distance(orig_features, cf_features, disc_mask=None, min_range=None, max_range=None):
    """Calculate various distance metrics between original and CF features."""
    if orig_features is None or cf_features is None:
        return None
    
    # Convert to torch tensors if they're numpy arrays
    if isinstance(orig_features, np.ndarray):
        orig_features = torch.tensor(orig_features)
        cf_features = torch.tensor(cf_features)
    
    # Ensure tensors are on the same device
    if disc_mask is not None:
        device = disc_mask.device
        orig_features = orig_features.to(device)
        cf_features = cf_features.to(device)
    
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
    
    # Calculate HEOM distance if masks are provided
    heom_dist = None
    if disc_mask is not None and min_range is not None and max_range is not None:
        heom_dist = heom_distance(
            orig_features, 
            cf_features, 
            disc_mask,
            (min_range, max_range)
        ).item()
    
    return {
        "l1_distance": l1_dist,
        "l2_distance": l2_dist,
        "cosine_distance": cos_dist,
        "hamming_distance": hamming,
        "heom_distance": heom_dist
    }

def fidelity(oracle, node_idx, cf_graph, full_graph, y_true, target_local_idx, target_class):
    """
    Calculate fidelity metrics for counterfactual evaluation.
    """
    oracle.eval()
    with torch.no_grad():
        # Ensure consistent device
        device = full_graph.x.device
        cf_graph = cf_graph.to(device)
        
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

def calculate_sparsity_metrics(feature_changes, edge_changes, total_edges, n_features):
    """Calculate detailed sparsity metrics."""
    metrics = {}
    
    # Node sparsity
    if feature_changes is not None:
        metrics["node_sparsity"] = {
            "absolute": feature_changes,
            "relative": feature_changes / n_features if n_features > 0 else 0
        }
    
    # Edge sparsity - defined as the fraction of edges touching the target node that were removed
    if edge_changes is not None and total_edges is not None:
        metrics["edge_sparsity"] = {
            "absolute": edge_changes,
            "relative": edge_changes / total_edges if total_edges > 0 else 0
        }
    
    return metrics

def graph_embedding_distance(orig_graph, cf_graph, oracle):
    """Calculate the L2 distance between graph embeddings"""
    oracle.eval()
    with torch.no_grad():
        # Ensure consistent device
        device = next(oracle.parameters()).device
        orig_graph = orig_graph.to(device)
        cf_graph = cf_graph.to(device)
        
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
        # Ensure consistent device
        device = next(oracle.parameters()).device
        full_graph = full_graph.to(device)
        
        # Get hidden activations from first GCNConv for the full graph
        h_full = oracle.conv1(full_graph.x, full_graph.edge_index)
        
        # Mean-pool to get the dataset's mean embedding
        mean_embedding = h_full.mean(dim=0)
        
        return mean_embedding

def calculate_distribution_distance(graph, oracle, dataset_mean_embedding):
    """Calculate the L2 distance between a graph's embedding and the dataset's mean embedding"""
    oracle.eval()
    with torch.no_grad():
        # Ensure consistent device
        device = next(oracle.parameters()).device
        graph = graph.to(device)
        dataset_mean_embedding = dataset_mean_embedding.to(device)
        
        # Get hidden activations from first GCNConv
        h_graph = oracle.conv1(graph.x, graph.edge_index)
        
        # Mean-pool to get graph embedding
        graph_embedding = h_graph.mean(dim=0)
        
        # Return L2 distance to the dataset mean
        return torch.norm(graph_embedding - dataset_mean_embedding, p=2).item()

def load_data(dataset_name, device):
    """Load dataset with proper preprocessing."""
    if dataset_name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root='./data/' + dataset_name, name=dataset_name)
        data = dataset[0].to(device)
        data.edge_index = to_undirected(data.edge_index).to(device)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        # WebKB datasets
        dataset = WebKB(root='./data/WebKB', name=dataset_name)
        data = dataset[0].to(device)
        data.edge_index = to_undirected(data.edge_index).to(device)
        
        # WebKB datasets come with 10 different train/val/test splits for cross-validation
        # We'll use the first split (index 0)
        split_idx = 0
        
        # Convert the multi-split masks to single boolean masks
        num_nodes = data.x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        
        # Use the first split (column 0) of each mask
        train_mask[data.train_mask[:, split_idx]] = True
        val_mask[data.val_mask[:, split_idx]] = True
        test_mask[data.test_mask[:, split_idx]] = True
        
        # Replace the multi-split masks with single boolean masks
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
    else:
        print(f"Unsupported dataset: {dataset_name}")
        sys.exit(1)
    
    return data

def train_oracle(model, g, epochs=500, lr=1e-2, wd=5e-4):
    """Train the GCN oracle model."""
    print(f"Training oracle model...")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = F.nll_loss
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(g.x, g.edge_index)
        loss = loss_fn(out[g.train_mask], g.y[g.train_mask])
        loss.backward()
        opt.step()
        
        # Print progress every 100 epochs
        if (epoch+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_logits = model(g.x, g.edge_index)
                val_acc = (val_logits.argmax(1)[g.val_mask] == g.y[g.val_mask]).float().mean().item()
                print(f"Epoch {epoch+1}/{epochs}: Val accuracy = {val_acc:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        logits = model(g.x, g.edge_index)
        acc = (logits.argmax(1)[g.test_mask] == g.y[g.test_mask]).float().mean().item()
    print(f"Oracle test accuracy: {acc:.4f}")
    return acc

def k_hop_graph(data, node_idx, k, return_sub_data=False):
    """Extract k-hop subgraph around a target node."""
    device = data.x.device
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, k, data.edge_index, relabel_nodes=True,
        num_nodes=data.num_nodes, flow='source_to_target'
    )
    x = data.x[subset]
    y = data.y[subset]
    sub_data = Data(x=x, edge_index=edge_index, y=y)
    sub_data.node_mask = subset
    sub_data.node_idx = mapping.item()  # target node in the subgraph
    return_val = (subset, edge_index, mapping) if not return_sub_data else sub_data
    return return_val

def sync_device(tensor, device):
    """Utility function to move tensors to the specified device safely."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    return tensor

def evaluate_explainer(method_name, data, oracle_model, node_idx, hops, device, n_features, disc_mask=None, min_range=None, max_range=None):
    """
    Generate and evaluate a counterfactual explanation using the specified method.
    """
    start_time = time.time()
    
    # Get k-hop subgraph
    subgraph_data = k_hop_graph(data, node_idx, hops, return_sub_data=True)
    subgraph_node_idx = subgraph_data.node_idx
    
    # Make sure subgraph data is on the right device
    subgraph_data = subgraph_data.to(device)
    
    # Get original prediction and class probabilities
    with torch.no_grad():
        logits = oracle_model(data.x, data.edge_index)
        orig_pred = logits[node_idx].argmax().item()
        orig_probs = F.softmax(logits[node_idx], dim=0)
        
        # Select target class as 2nd highest probability
        target_logits = logits[node_idx].clone()
        target_logits[orig_pred] = float('-inf')
        target_class = target_logits.argmax().item()
        
        # Also get the subgraph prediction (for debugging)
        subgraph_pred = oracle_model(subgraph_data.x, subgraph_data.edge_index)[subgraph_node_idx].argmax().item()
    
    # Initialize explainer based on method name
    try:
        # Force model and tensors to same device to prevent device mismatch
        oracle_model = oracle_model.to(device)
        subgraph_data = subgraph_data.to(device)
        
        if method_name == 'ego':
            explainer = EGOExplainer(
                model=oracle_model,
                node_idx=subgraph_node_idx,
                edge_index=subgraph_data.edge_index,
                features=subgraph_data.x,
                num_classes=data.y.max().item() + 1,
                target=target_class,
                hops=2,
                max_nodes=20
            )
            cf_edge_index, cf_features, cf_node_idx, original_indices = explainer.explain()
            
            # Ensure tensors are on the correct device
            cf_edge_index = sync_device(cf_edge_index, device)
            cf_features = sync_device(cf_features, device)
            if isinstance(cf_node_idx, torch.Tensor):
                cf_node_idx = sync_device(cf_node_idx, device)
            original_indices = sync_device(original_indices, device)
        
        elif method_name == 'cff':
            explainer = CFFExplainer(
                model=oracle_model,
                node_idx=subgraph_node_idx,
                edge_index=subgraph_data.edge_index,
                features=subgraph_data.x,
                num_classes=data.y.max().item() + 1,
                target=target_class,
                n_trials=100,
                alpha=0.5
            )
            cf_edge_index, cf_features, cf_node_idx, original_indices = explainer.explain()
            
            # Ensure tensors are on the correct device
            cf_edge_index = sync_device(cf_edge_index, device)
            cf_features = sync_device(cf_features, device)
            if isinstance(cf_node_idx, torch.Tensor):
                cf_node_idx = sync_device(cf_node_idx, device)
            original_indices = sync_device(original_indices, device)
        
        elif method_name == 'cf_gnn':
            explainer = CFGNNExplainer(
                model=oracle_model,
                node_idx=subgraph_node_idx,
                edge_index=subgraph_data.edge_index,
                features=subgraph_data.x,
                target=target_class,
                epochs=200
            )
            cf_edge_index, cf_features = explainer.explain()
            
            # Ensure tensors are on the correct device
            cf_edge_index = sync_device(cf_edge_index, device)
            cf_features = sync_device(cf_features, device)
            cf_node_idx = subgraph_node_idx  # Doesn't change in CF-GNN
            original_indices = torch.arange(subgraph_data.x.shape[0], device=device)
        
        elif method_name == 'random':
            explainer = RandomExplainer(
                model=oracle_model,
                node_idx=subgraph_node_idx,
                edge_index=subgraph_data.edge_index,
                features=subgraph_data.x,
                num_classes=data.y.max().item() + 1,
                target=target_class,
                max_iters=100,
                edge_prob=0.5,
                feature_prob=0.3
            )
            cf_edge_index, cf_features, cf_node_idx, original_indices = explainer.explain()
            
            # Ensure tensors are on the correct device
            cf_edge_index = sync_device(cf_edge_index, device)
            cf_features = sync_device(cf_features, device)
            if isinstance(cf_node_idx, torch.Tensor):
                cf_node_idx = sync_device(cf_node_idx, device)
            original_indices = sync_device(original_indices, device)
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
            
    except Exception as e:
        print(f"Error in {method_name} explainer: {e}")
        return {
            "success": False,
            "error": str(e),
            "time_seconds": time.time() - start_time
        }
        
    # Measure elapsed time
    elapsed_time = time.time() - start_time
    
    # Create counterfactual graph and evaluate
    cf_graph = Data(x=cf_features, edge_index=cf_edge_index).to(device)
    
    # Evaluate counterfactual prediction
    with torch.no_grad():
        cf_pred = oracle_model(cf_features, cf_edge_index)[cf_node_idx].argmax().item()
    
    # Calculate feature changes
    orig_feature = subgraph_data.x[subgraph_node_idx]
    cf_feature = cf_features[cf_node_idx]
    feature_changes = (orig_feature != cf_feature).sum().item()
    
    # Calculate edge changes - ensure indices are integers for set operations
    original_edges = set([(subgraph_data.edge_index[0, i].item(), subgraph_data.edge_index[1, i].item()) 
                        for i in range(subgraph_data.edge_index.shape[1])])
    cf_edges = set([(cf_edge_index[0, i].item(), cf_edge_index[1, i].item()) 
                    for i in range(cf_edge_index.shape[1])])
    
    edges_added = len(cf_edges - original_edges)
    edges_removed = len(original_edges - cf_edges)
    total_edge_changes = edges_added + edges_removed
    total_edges = len(original_edges)  # Original number of edges
    
    # Calculate nodes retained
    num_nodes_retained = cf_features.shape[0]
    num_edges_retained = cf_edge_index.shape[1]
    num_original_nodes = subgraph_data.x.shape[0]
    num_original_edges = subgraph_data.edge_index.shape[1]
    
    # Calculate percentage changes
    nodes_removed_pct = (num_original_nodes - num_nodes_retained) / num_original_nodes * 100
    edges_removed_pct = (num_original_edges - num_edges_retained) / max(1, num_original_edges) * 100
    
    # Check if prediction flipped
    success = (cf_pred == target_class)
    
    # Calculate fidelity metrics
    ground_truth = data.y[node_idx].item()
    fidelity_metrics = fidelity(oracle_model, node_idx, cf_graph, data, ground_truth, cf_node_idx, target_class)
    
    # Calculate feature metrics
    feature_metrics = calculate_feature_distance(orig_feature, cf_feature, disc_mask, min_range, max_range)
    
    # Calculate sparsity metrics
    sparsity_metrics = calculate_sparsity_metrics(feature_changes, total_edge_changes, total_edges, n_features)
    
    # Calculate graph embedding distance
    # Create factual graph for embedding distance calculation
    factual_graph = Data(x=subgraph_data.x, edge_index=subgraph_data.edge_index)
    embedding_distance = graph_embedding_distance(factual_graph, cf_graph, oracle_model)
    
    # Calculate distribution distance
    dataset_mean_embedding = calculate_dataset_mean_embedding(data, oracle_model)
    distribution_distance = calculate_distribution_distance(cf_graph, oracle_model, dataset_mean_embedding)
    
    # Prepare results dictionary with all metrics
    results = {
        "success": success,
        "method": method_name,
        "node_idx": node_idx,
        "original_prediction": orig_pred,
        "original_subgraph_prediction": subgraph_pred,
        "target_prediction": target_class, 
        "cf_prediction": cf_pred,
        "feature_changes": feature_changes,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
        "total_edge_changes": total_edge_changes,
        "total_edges": total_edges,
        "nodes_removed_pct": nodes_removed_pct,
        "edges_removed_pct": edges_removed_pct,
        "original_nodes": num_original_nodes,
        "original_edges": num_original_edges,
        "cf_nodes": num_nodes_retained,
        "cf_edges": num_edges_retained,
        "time_seconds": elapsed_time,
        "fidelity_metrics": fidelity_metrics,
        "feature_metrics": feature_metrics,
        "sparsity_metrics": sparsity_metrics,
        "embedding_distance": embedding_distance,
        "distribution_distance": distribution_distance
    }
    
    return results

def run_evaluation(dataset_name, methods, num_samples=10, hops=2, seed=42, model_type="gcn", cuda_device=0):
    """
    Run comprehensive evaluation of multiple counterfactual explanation methods.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Use specified CUDA device if available
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_device}')
        print(f"Using CUDA device: {cuda_device}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    data = load_data(dataset_name, device)
    n_features = data.x.shape[1]
    n_classes = data.y.max().item() + 1
    
    # Move everything to the same device to prevent device mismatch
    data = data.to(device)  
    
    # Identify binary/continuous features
    is_binary = ((data.x == 0) | (data.x == 1)).all(dim=0)
    disc_mask = is_binary
    
    # Feature bounds
    min_range = torch.zeros(n_features, device=device)
    max_range = torch.ones(n_features, device=device)
    
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {data.num_nodes} | Edges: {data.num_edges//2} | Features: {n_features}")
    print(f"Classes: {n_classes} | Binary features: {disc_mask.sum().item()}/{n_features}")
    
    # Create and train oracle model based on model_type
    if model_type.lower() == "chebnet":
        print("Using ChebNet model as oracle")
        oracle_model = ChebNet(n_features, 16, n_classes).to(device)
    else:
        print("Using GCN model as oracle")
        oracle_model = GCN(n_features, 16, n_classes).to(device)
    
    train_oracle(oracle_model, data)
    
    # Make directory for results
    os.makedirs("results", exist_ok=True)
    
    # Test nodes selection - randomly select from test mask
    test_indices = (data.test_mask == 1).nonzero().view(-1)
    if len(test_indices) > num_samples:
        selected_indices = test_indices[torch.randperm(len(test_indices))[:num_samples]]
    else:
        selected_indices = test_indices
    
    print(f"Selected {len(selected_indices)} test nodes")
    
    # Store results for each method
    all_results = {method: [] for method in methods}
    
    # Evaluate each method on each node
    for i, node_idx in enumerate(selected_indices):
        node_idx = node_idx.item()
        print(f"\nEvaluating node {i+1}/{len(selected_indices)} (idx: {node_idx})")
        
        for method in methods:
            print(f"  Running {method.upper()} method...")
            result = evaluate_explainer(
                method_name=method,
                data=data,
                oracle_model=oracle_model,
                node_idx=node_idx,
                hops=hops,
                device=device,
                n_features=n_features,
                disc_mask=disc_mask,
                min_range=min_range,
                max_range=max_range
            )
            
            all_results[method].append(result)
            
            # Print summary
            if result["success"]:
                print(f"    Success! {result['original_prediction']} → {result['target_prediction']} in {result['time_seconds']:.2f}s")
                print(f"    Features: {result['feature_changes']} changes | Edges: +{result['edges_added']}, -{result['edges_removed']}")
            else:
                if "error" in result:
                    print(f"    Failed with error: {result['error']}")
                else:
                    print(f"    Failed to find a valid counterfactual in {result['time_seconds']:.2f}s")
    
    # Calculate summary stats for each method
    summary = {}
    for method in methods:
        results = all_results[method]
        
        # Get successful results only
        successful_results = [r for r in results if r["success"]]
        success_rate = len(successful_results) / len(results) * 100 if results else 0
        
        # Calculate metrics averages
        if successful_results:
            # Feature and edge changes
            avg_feature_changes = np.mean([r.get("feature_changes", 0) for r in successful_results])
            avg_edges_added = np.mean([r.get("edges_added", 0) for r in successful_results])
            avg_edges_removed = np.mean([r.get("edges_removed", 0) for r in successful_results])
            
            # Sparsity
            avg_node_sparsity = np.mean([r.get("sparsity_metrics", {}).get("node_sparsity", {}).get("relative", 0) 
                                       for r in successful_results])
            avg_edge_sparsity = np.mean([r.get("sparsity_metrics", {}).get("edge_sparsity", {}).get("relative", 0) 
                                       for r in successful_results])
            
            # Fidelity
            avg_flipped = np.mean([r.get("fidelity_metrics", {}).get("flipped", 0) for r in successful_results])
            avg_psi = np.mean([r.get("fidelity_metrics", {}).get("psi", 0) for r in successful_results])
            avg_valid_simple = np.mean([r.get("fidelity_metrics", {}).get("valid_simple", 0) for r in successful_results])
            
            # Distances
            avg_l1 = np.mean([r.get("feature_metrics", {}).get("l1_distance", 0) for r in successful_results])
            avg_l2 = np.mean([r.get("feature_metrics", {}).get("l2_distance", 0) for r in successful_results])
            avg_cosine = np.mean([r.get("feature_metrics", {}).get("cosine_distance", 0) for r in successful_results])
            avg_hamming = np.mean([r.get("feature_metrics", {}).get("hamming_distance", 0) for r in successful_results])
            avg_heom = np.mean([r.get("feature_metrics", {}).get("heom_distance", 0) for r in successful_results if "heom_distance" in r.get("feature_metrics", {})])
            
            # Embedding and distribution distances
            avg_embedding = np.mean([r.get("embedding_distance", 0) for r in successful_results])
            avg_distribution = np.mean([r.get("distribution_distance", 0) for r in successful_results])
        else:
            # Set defaults if no successful results
            avg_feature_changes = 0
            avg_edges_added = 0
            avg_edges_removed = 0
            avg_node_sparsity = 0
            avg_edge_sparsity = 0
            avg_flipped = 0
            avg_psi = 0
            avg_valid_simple = 0
            avg_l1 = 0
            avg_l2 = 0
            avg_cosine = 0
            avg_hamming = 0
            avg_heom = 0
            avg_embedding = 0
            avg_distribution = 0
        
        # Average time (for all attempts, successful or not)
        avg_time = np.mean([r.get("time_seconds", 0) for r in results])
        
        # Store summary metrics
        summary[method] = {
            "success_rate": success_rate,
            "avg_feature_changes": avg_feature_changes,
            "avg_edges_added": avg_edges_added,
            "avg_edges_removed": avg_edges_removed,
            "avg_node_sparsity": avg_node_sparsity,
            "avg_edge_sparsity": avg_edge_sparsity,
            "avg_flipped": avg_flipped,
            "avg_psi": avg_psi,
            "avg_valid_simple": avg_valid_simple,
            "avg_l1_distance": avg_l1,
            "avg_l2_distance": avg_l2,
            "avg_cosine_distance": avg_cosine,
            "avg_hamming_distance": avg_hamming,
            "avg_heom_distance": avg_heom,
            "avg_embedding_distance": avg_embedding,
            "avg_distribution_distance": avg_distribution,
            "avg_time": avg_time
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/cf_comparison_{dataset_name}_{timestamp}.pt"
    torch.save({
        "dataset": dataset_name,
        "methods": methods,
        "results": all_results,
        "summary": summary
    }, results_path)
    
    # Also save as CSV for easy analysis
    for method in methods:
        # Convert results to DataFrame
        df = pd.DataFrame(all_results[method])
        df.to_csv(f"results/{method}_{dataset_name}_results.csv", index=False)
    
    # Print summary table
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY FOR {dataset_name.upper()}")
    print("="*80)
    
    print(f"{'Method':<10} | {'Success':<10} | {'Feat Δ':<10} | {'Edge Δ':<10} | {'Ψ':<5} | {'Time (s)':<10}")
    print("-"*70)
    
    for method in methods:
        s = summary[method]
        print(f"{method:<10} | {s['success_rate']:>8.1f}% | {s['avg_feature_changes']:>8.2f} | {s['avg_edges_added']+s['avg_edges_removed']:>8.2f} | {s['avg_psi']:>3.1f} | {s['avg_time']:>8.2f}")
    
    print("\nDetailed metrics saved to:", results_path)
    
    return all_results, summary

def main():
    parser = argparse.ArgumentParser(description='Counterfactual Explainer Evaluation')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use (Cora, Citeseer, Cornell, etc.)')
    parser.add_argument('--methods', type=str, nargs='+', default=['ego', 'cff', 'cf_gnn', 'random'], 
                       help='Methods to evaluate (ego, cff, cf_gnn, random)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples to explain')
    parser.add_argument('--hops', type=int, default=2, help='Number of hops for subgraph extraction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--model', type=str, default='gcn', help='Model type (gcn or chebnet)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device to use (e.g., 0, 1, 2)')
    
    args = parser.parse_args()
    
    # Run evaluation
    start_time = time.time()
    results, summary = run_evaluation(
        dataset_name=args.dataset,
        methods=args.methods,
        num_samples=args.samples,
        hops=args.hops,
        seed=args.seed,
        model_type=args.model,
        cuda_device=args.cuda
    )
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f}s")
    
    # Create visualizations if requested
    if args.visualize:
        try:
            import matplotlib.pyplot as plt
            # Bar chart of success rates
            plt.figure(figsize=(10, 6))
            methods = list(summary.keys())
            success_rates = [summary[m]["success_rate"] for m in methods]
            
            plt.bar(methods, success_rates)
            plt.xlabel("Method")
            plt.ylabel("Success Rate (%)")
            plt.title(f"Counterfactual Success Rates on {args.dataset}")
            plt.ylim(0, 100)
            
            plt.savefig(f"results/success_rates_{args.dataset}.png")
            print(f"Created visualization: results/success_rates_{args.dataset}.png")
            
            # Performance metrics
            plt.figure(figsize=(12, 8))
            metrics = ["avg_feature_changes", "avg_edges_added", "avg_edges_removed", 
                      "avg_node_sparsity", "avg_edge_sparsity", "avg_embedding_distance", "avg_time"]
            metric_labels = ["Feature Changes", "Edges Added", "Edges Removed", 
                           "Node Sparsity", "Edge Sparsity", "Embedding Distance", "Time (s)"]
            
            cols = 3
            rows = (len(metrics) + cols - 1) // cols
            
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                plt.subplot(rows, cols, i+1)
                values = [summary[m][metric] for m in methods]
                plt.bar(methods, values)
                plt.title(label)
                
            plt.tight_layout()
            plt.savefig(f"results/metrics_{args.dataset}.png")
            print(f"Created visualization: results/metrics_{args.dataset}.png")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    main() 