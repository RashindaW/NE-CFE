import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import numpy as np
import argparse
import time
import os

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K=3):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, hidden_channels, K=K)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=K)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def k_hop_graph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None):
    # Extract the k-hop subgraph of node_idx from edge_index
    device = edge_index.device
    
    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    
    # Convert to CPU for subgraph extraction to avoid CUDA issues
    cpu_edge_index = edge_index.cpu()
    
    # Compute k-hop subgraph
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx, 
        num_hops, 
        cpu_edge_index, 
        relabel_nodes=relabel_nodes, 
        num_nodes=num_nodes
    )
    
    # Move back to original device
    subset = subset.to(device)
    sub_edge_index = sub_edge_index.to(device) if sub_edge_index.numel() > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
    mapping = mapping.to(device)
    
    return subset, sub_edge_index, mapping

class EGOExplainer:
    """
    Implementation of the EGO method for counterfactual explanations
    """
    def __init__(self, model, edge_index, features, node_idx, num_classes, target=None, hops=2, max_nodes=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.edge_index = edge_index.to(self.device)
        self.features = features.to(self.device)
        self.node_idx = node_idx
        self.num_classes = num_classes
        self.hops = hops
        self.max_nodes = max_nodes
        
        with torch.no_grad():
            self.original_pred = model(self.features, self.edge_index)[self.node_idx].argmax().item()
        
        self.target = target if target is not None else 1 - self.original_pred
    
    def _retrieve_subgraph(self):
        """Extract k-hop subgraph around the node"""
        subset, sub_edge_index, mapping = k_hop_graph(
            node_idx=self.node_idx,
            num_hops=self.hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.features.shape[0]
        )
        
        # Get node index in the subgraph
        sub_node_idx = mapping.item()
        
        # Get subgraph features
        sub_features = self.features[subset].clone()
        
        return subset, sub_edge_index, sub_features, sub_node_idx
    
    def _modify_features(self, features, sub_edge_index, sub_node_idx):
        """Modify features to create counterfactual explanation"""
        # Get the current prediction
        with torch.no_grad():
            logits = self.model(features, sub_edge_index)
            current_pred = logits[sub_node_idx].argmax().item()
        
        # If already predicted as target, no need to modify
        if current_pred == self.target:
            return features
        
        # Create a mask of nodes per class (excluding the target node)
        class_masks = {}
        for c in range(self.num_classes):
            with torch.no_grad():
                class_mask = (logits.argmax(dim=1) == c)
                class_mask[sub_node_idx] = False  # Exclude the target node
                class_masks[c] = class_mask
        
        # Create a modified version of the features
        modified_features = features.clone()
        
        # Find nodes predicted as the target class
        target_nodes = class_masks.get(self.target, torch.zeros(features.size(0), dtype=torch.bool, device=self.device))
        
        # If there are nodes predicted as target class, use their features
        if target_nodes.sum() > 0:
            # Calculate mean features of target class nodes
            target_features = features[target_nodes].mean(dim=0)
            
            # Update features of the target node to be similar to target class nodes
            modified_features[sub_node_idx] = target_features
        else:
            # If no target class nodes found, try to move away from current class
            original_class_nodes = class_masks.get(current_pred, torch.zeros(features.size(0), dtype=torch.bool, device=self.device))
            
            if original_class_nodes.sum() > 0:
                # Calculate mean features of original class
                original_features = features[original_class_nodes].mean(dim=0)
                
                # Move away from original class (add noise or invert direction)
                modified_features[sub_node_idx] = features[sub_node_idx] - (original_features - features[sub_node_idx])
        
        return modified_features
    
    def explain(self):
        """Generate counterfactual explanation"""
        # Extract k-hop subgraph
        subset, sub_edge_index, sub_features, sub_node_idx = self._retrieve_subgraph()
        
        # Check if prediction is already the target
        with torch.no_grad():
            sub_pred = self.model(sub_features, sub_edge_index)[sub_node_idx].argmax().item()
        
        if sub_pred == self.target:
            return sub_edge_index, sub_features, sub_node_idx, subset
        
        # Iteratively modify features until target prediction or max iterations
        max_iterations = 10
        best_features = sub_features.clone()
        
        for i in range(max_iterations):
            # Modify features
            modified_features = self._modify_features(best_features, sub_edge_index, sub_node_idx)
            
            # Check if counterfactual is found
            with torch.no_grad():
                current_pred = self.model(modified_features, sub_edge_index)[sub_node_idx].argmax().item()
            
            if current_pred == self.target:
                best_features = modified_features
                break
            
            best_features = modified_features
        
        return sub_edge_index, best_features, sub_node_idx, subset

def load_data(dataset_name):
    """
    Load dataset using Planetoid
    """
    if dataset_name.lower() == 'cora':
        dataset = Planetoid(root='./data/Cora', name='Cora')
    elif dataset_name.lower() == 'citeseer':
        dataset = Planetoid(root='./data/Citeseer', name='Citeseer')
    elif dataset_name.lower() == 'pubmed':
        dataset = Planetoid(root='./data/PubMed', name='PubMed')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    data = dataset[0]
    num_classes = dataset.num_classes
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move data to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    
    return data, num_classes

def train_oracle_model(data, num_classes, hidden_channels=16, epochs=200):
    """
    Train a GCN model on the dataset
    """
    model = GCN(data.x.size(1), hidden_channels, num_classes).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    return model

def generate_counterfactuals_for_dataset(dataset_name, num_samples=10, hops=2):
    """
    Generate counterfactual explanations for a dataset using the EGO method
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    data, num_classes = load_data(dataset_name)
    print(f"Dataset: {dataset_name}, Num classes: {num_classes}")
    
    # Train oracle model
    model = train_oracle_model(data, num_classes)
    model.eval()
    
    # Evaluate model
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        acc = int(correct) / int(data.train_mask.sum())
        print(f"Oracle model accuracy: {acc:.4f}")
    
    # Create directory for results
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Collect stats
    stats = {
        'dataset': dataset_name,
        'model_acc': acc,
        'samples': [],
        'ego_found_cf': 0,
        'ego_time': []
    }
    
    # Generate counterfactuals for random nodes
    all_indices = torch.where(data.train_mask)[0]
    if len(all_indices) > num_samples:
        indices = all_indices[torch.randperm(len(all_indices))[:num_samples]]
    else:
        indices = all_indices
    
    print(f"Generating counterfactuals for {len(indices)} nodes")
    
    for i, node_idx in enumerate(indices):
        node_idx = node_idx.item()
        print(f"Processing node {i+1}/{len(indices)}, idx={node_idx}")
        
        # Get original prediction
        with torch.no_grad():
            original_pred = model(data.x, data.edge_index)[node_idx].argmax().item()
        
        # Target is the opposite class (for binary classification)
        target = 1 - original_pred if num_classes == 2 else (original_pred + 1) % num_classes
        
        # Save sample info
        sample_info = {
            'node_idx': node_idx,
            'original_pred': original_pred,
            'target': target
        }
        
        # Generate counterfactual using EGO
        ego_explainer = EGOExplainer(model, data.edge_index, data.x, node_idx, num_classes, target=target, hops=hops)
        
        try:
            # Time the explanation
            start_time = time.time()
            cf_edge_index, cf_features, cf_node_idx, cf_subset = ego_explainer.explain()
            end_time = time.time()
            
            # Check if counterfactual is valid
            with torch.no_grad():
                cf_pred = model(cf_features, cf_edge_index)[cf_node_idx].argmax().item()
            
            # Update stats
            ego_time = end_time - start_time
            ego_success = cf_pred == target
            
            if ego_success:
                stats['ego_found_cf'] += 1
            
            stats['ego_time'].append(ego_time)
            
            sample_info.update({
                'ego_success': ego_success,
                'ego_time': ego_time,
                'ego_pred': cf_pred
            })
            
            print(f"EGO - Success: {ego_success}, Time: {ego_time:.4f}s, CF Pred: {cf_pred}")
            
        except Exception as e:
            print(f"Error generating EGO explanation: {e}")
            sample_info.update({
                'ego_success': False,
                'ego_error': str(e)
            })
        
        stats['samples'].append(sample_info)
    
    # Save stats
    with open(f"{results_dir}/ego_stats.txt", 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model Accuracy: {acc:.4f}\n")
        f.write(f"EGO Success Rate: {stats['ego_found_cf']}/{len(indices)} ({stats['ego_found_cf']/len(indices):.4f})\n")
        if stats['ego_time']:
            f.write(f"EGO Avg Time: {sum(stats['ego_time'])/len(stats['ego_time']):.4f}s\n")
        
        f.write("\nSample Details:\n")
        for i, sample in enumerate(stats['samples']):
            f.write(f"Sample {i+1}:\n")
            for k, v in sample.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EGO counterfactual explanations")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of nodes to explain")
    parser.add_argument("--hops", type=int, default=2, help="Number of hops for subgraph extraction")
    args = parser.parse_args()
    
    generate_counterfactuals_for_dataset(args.dataset, args.num_samples, args.hops) 