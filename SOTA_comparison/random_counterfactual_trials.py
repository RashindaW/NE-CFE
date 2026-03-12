import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
import numpy as np
import sys
import time
import argparse
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pandas as pd
import os
import random

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

class RandomExplainer:
    """
    Implementation of a Random method for counterfactual explanations
    Simply makes random perturbations to edges and features until the prediction changes
    """
    def __init__(self, model, node_idx, edge_index, features, num_classes, 
                target=None, max_iters=100, edge_prob=0.5, feature_prob=0.3):
        self.model = model
        self.node_idx = node_idx
        # Get device from model
        self.device = next(model.parameters()).device
        self.edge_index = edge_index.clone().to(self.device)
        self.features = features.clone().to(self.device)
        self.num_classes = num_classes
        self.max_iters = max_iters
        self.edge_prob = edge_prob
        self.feature_prob = feature_prob
        
        with torch.no_grad():
            self.original_pred = model(self.features, self.edge_index)[self.node_idx].argmax().item()
        
        self.target = target if target is not None else 1 - self.original_pred
    
    def _random_edge_mask(self, num_edges):
        # Randomly mask edges with probability self.edge_prob
        return torch.tensor([random.random() > self.edge_prob for _ in range(num_edges)], dtype=torch.bool, device=self.device)
    
    def _random_feature_perturbation(self, features):
        # Create a clone of features
        perturbed_features = features.clone()
        
        # For the target node, randomly perturb features with probability self.feature_prob
        for i in range(perturbed_features[self.node_idx].shape[0]):
            if random.random() < self.feature_prob:
                # Add random noise to the feature
                perturbed_features[self.node_idx, i] += torch.randn(1, device=self.device).item() * 0.1
                
        return perturbed_features
    
    def explain(self):
        best_edge_index = self.edge_index.clone()
        best_features = self.features.clone()
        best_pred = self.original_pred
        
        # First try just perturbing features without changing the graph structure
        for _ in range(self.max_iters // 4):
            perturbed_features = self._random_feature_perturbation(self.features)
            
            with torch.no_grad():
                pred = self.model(perturbed_features, self.edge_index)[self.node_idx].argmax().item()
            
            if pred == self.target:
                return self.edge_index, perturbed_features, self.node_idx, torch.arange(self.features.shape[0], device=self.device)
        
        # If feature perturbation alone doesn't work, try perturbing both edges and features
        for i in range(self.max_iters):
            # Create edge mask
            edge_mask = self._random_edge_mask(self.edge_index.shape[1])
            perturbed_edge_index = self.edge_index[:, edge_mask]
            
            # Perturb features
            perturbed_features = self._random_feature_perturbation(self.features)
            
            # Check if the graph is still connected to the target node
            # At least one edge must connect to the target node
            target_connected = False
            for j in range(perturbed_edge_index.shape[1]):
                if perturbed_edge_index[0, j].item() == self.node_idx or perturbed_edge_index[1, j].item() == self.node_idx:
                    target_connected = True
                    break
            
            if not target_connected:
                continue
            
            with torch.no_grad():
                pred = self.model(perturbed_features, perturbed_edge_index)[self.node_idx].argmax().item()
            
            if pred == self.target:
                return perturbed_edge_index, perturbed_features, self.node_idx, torch.arange(self.features.shape[0], device=self.device)
        
        # If we exhausted all iterations and still didn't find a counterfactual,
        # try one more with more aggressive perturbation
        edge_mask = torch.tensor([random.random() > 0.7 for _ in range(self.edge_index.shape[1])], dtype=torch.bool, device=self.device)
        perturbed_edge_index = self.edge_index[:, edge_mask]
        
        perturbed_features = self.features.clone()
        # More aggressive feature perturbation for the target node
        perturbed_features[self.node_idx] = perturbed_features[self.node_idx] + torch.randn_like(perturbed_features[self.node_idx], device=self.device) * 0.2
        
        with torch.no_grad():
            pred = self.model(perturbed_features, perturbed_edge_index)[self.node_idx].argmax().item()
        
        if pred == self.target:
            return perturbed_edge_index, perturbed_features, self.node_idx, torch.arange(self.features.shape[0], device=self.device)
        
        # If still no success, return the original graph
        # This indicates failure to find a counterfactual
        return self.edge_index, self.features, self.node_idx, torch.arange(self.features.shape[0], device=self.device)

def load_data(dataset_name):
    if dataset_name in ['Cora', 'Citeseer']:
        dataset = Planetoid(root='./data/' + dataset_name, name=dataset_name)
        data = dataset[0]
    elif dataset_name == 'AIDS':
        # This is a placeholder for AIDS dataset loading
        print("AIDS dataset loading not implemented yet")
        sys.exit(1)
    elif dataset_name == 'Facebook':
        # This is a placeholder for Facebook dataset loading
        print("Facebook dataset loading not implemented yet")
        sys.exit(1)
    else:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    return data

def train_oracle(model, g, epochs=400, lr=1e-2, wd=5e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); loss_fn = F.nll_loss
    for _ in range(epochs):
        model.train(); opt.zero_grad(); out = model(g.x, g.edge_index); loss = loss_fn(out[g.train_mask], g.y[g.train_mask]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(g.x, g.edge_index); acc = (logits.argmax(1)[g.test_mask] == g.y[g.test_mask]).float().mean().item()
    return acc

def k_hop_graph(data, node_idx, k, return_sub_data=False):
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

def train_and_generate_cf_for_dataset(dataset_name, num_samples=10, hops=2):
    data = load_data(dataset_name)
    hidden = 16; out = data.y.max().item() + 1
    
    # Train oracle model
    oracle_model = GCN(data.x.shape[1], hidden, out)
    accuracy = train_oracle(oracle_model, data)
    print(f"Oracle accuracy: {accuracy:.4f}")
    
    oracle_model.eval()
    
    results = []
    
    # Test nodes selection - randomly select from test mask
    test_indices = (data.test_mask == 1).nonzero().view(-1)
    if len(test_indices) > num_samples:
        selected_indices = test_indices[torch.randperm(len(test_indices))[:num_samples]]
    else:
        selected_indices = test_indices
    
    print(f"Selected {len(selected_indices)} test nodes")
    
    for i, node_idx in enumerate(selected_indices):
        node_idx = node_idx.item()
        
        # Get k-hop subgraph
        subgraph_data = k_hop_graph(data, node_idx, hops, return_sub_data=True)
        subgraph_node_idx = subgraph_data.node_idx
        
        # Get original prediction
        with torch.no_grad():
            original_pred = oracle_model(data.x, data.edge_index)[node_idx].argmax().item()
            subgraph_pred = oracle_model(subgraph_data.x, subgraph_data.edge_index)[subgraph_node_idx].argmax().item()
        
        # Set target class (opposite of original)
        target_class = 1 - original_pred if original_pred < 2 else (original_pred + 1) % out
        
        # Time the counterfactual generation
        start_time = time.time()
        
        # Generate counterfactual explanation
        random_explainer = RandomExplainer(
            model=oracle_model,
            node_idx=subgraph_node_idx,
            edge_index=subgraph_data.edge_index,
            features=subgraph_data.x,
            num_classes=out,
            target=target_class,
            max_iters=100,
            edge_prob=0.5,
            feature_prob=0.3
        )
        
        cf_edge_index, cf_features, cf_node_idx, original_indices = random_explainer.explain()
        
        # Measure time
        elapsed_time = time.time() - start_time
        
        # Evaluate counterfactual
        with torch.no_grad():
            cf_pred = oracle_model(cf_features, cf_edge_index)[cf_node_idx].argmax().item()
            
        # Calculate number of nodes retained
        num_nodes_retained = cf_features.shape[0]
        num_edges_retained = cf_edge_index.shape[1]
        num_original_nodes = subgraph_data.x.shape[0]
        num_original_edges = subgraph_data.edge_index.shape[1]
        
        # Calculate changes as percentage of original
        nodes_removed_pct = (num_original_nodes - num_nodes_retained) / num_original_nodes * 100
        edges_removed_pct = (num_original_edges - num_edges_retained) / max(1, num_original_edges) * 100
        
        # Check if prediction flipped
        success = (cf_pred == target_class)
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'node_idx': node_idx,
            'original_prediction': original_pred,
            'target_prediction': target_class, 
            'cf_prediction': cf_pred,
            'success': success,
            'nodes_removed_pct': nodes_removed_pct,
            'edges_removed_pct': edges_removed_pct,
            'original_nodes': num_original_nodes,
            'original_edges': num_original_edges,
            'cf_nodes': num_nodes_retained,
            'cf_edges': num_edges_retained,
            'time_seconds': elapsed_time
        })
        
        print(f"Node {i+1}/{len(selected_indices)}: {'Success' if success else 'Failed'} in {elapsed_time:.2f}s")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(f'results/random_{dataset_name}_results.csv', index=False)
    
    # Print summary
    success_rate = results_df['success'].mean() * 100
    avg_nodes_removed = results_df['nodes_removed_pct'].mean()
    avg_edges_removed = results_df['edges_removed_pct'].mean()
    avg_time = results_df['time_seconds'].mean()
    
    print(f"\nResults for {dataset_name} dataset:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Nodes Removed: {avg_nodes_removed:.2f}%")
    print(f"Average Edges Removed: {avg_edges_removed:.2f}%")
    print(f"Average Time: {avg_time:.2f}s")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Random Explainer for counterfactual explanations')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use (Cora, Citeseer, AIDS, Facebook)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples to explain')
    parser.add_argument('--hops', type=int, default=2, help='Number of hops for subgraph extraction')
    
    args = parser.parse_args()
    
    train_and_generate_cf_for_dataset(args.dataset, args.samples, args.hops)

if __name__ == "__main__":
    main() 