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

class ExplainModelNodeMulti(torch.nn.Module):
    """
    Model wrapper for learning counterfactual explanations
    """
    def __init__(self, model, node_idx, edge_index, features):
        super(ExplainModelNodeMulti, self).__init__()
        self.model = model
        self.node_idx = node_idx
        self.edge_index = edge_index
        self.features = features
        self.num_edges = edge_index.size(1)
        
        # Get device from model
        self.device = next(model.parameters()).device
        
        # Learnable edge mask
        self.edge_mask = torch.nn.Parameter(torch.randn(self.num_edges, device=self.device))

    def forward(self):
        # Get masked adjacency matrix
        masked_adj = self.get_masked_adj()
        
        # Forward pass on counterfactual graph
        S_f = self.model(self.features, masked_adj[0], masked_adj[1])
        
        # Forward pass on original graph
        S_c = self.model(self.features, self.edge_index)
        
        return S_f, S_c

    def loss(self, S_f, S_c, pred_label, target_label, gamma=1.0, lambda_param=1.0, alpha=0.5):
        relu = torch.nn.ReLU()

        # Sort logits for counterfactual graph
        _, sorted_indices_f = torch.sort(S_f, descending=True)
        
        # Sort logits for original graph
        _, sorted_indices_c = torch.sort(S_c, descending=True)
        
        # For counterfactual: encourage target class prediction
        # If target class is not the top prediction, penalize
        if sorted_indices_f[0] != target_label:
            top_class_f = sorted_indices_f[0]
            L_f = relu(gamma + S_f[top_class_f] - S_f[target_label])
        else:
            L_f = torch.tensor(0.0, device=self.device)
        
        # For original: preserve original prediction
        # If original class is not the top prediction, penalize
        if sorted_indices_c[0] != pred_label:
            top_class_c = sorted_indices_c[0]
            L_c = relu(gamma + S_c[top_class_c] - S_c[pred_label])
        else:
            L_c = torch.tensor(0.0, device=self.device)

        # Encourage sparsity in edge mask
        edge_mask = self.get_edge_mask()
        L1 = torch.linalg.norm(edge_mask, ord=1)

        # Total loss
        loss = L1 + lambda_param * (alpha * L_f + (1 - alpha) * L_c)
        return loss

    def get_masked_adj(self):
        # Apply sigmoid to edge mask to constrain values between 0 and 1
        edge_weights = torch.sigmoid(self.edge_mask)
        
        # Return edge index and weights (for edge_weight parameter in GNN)
        return self.edge_index, edge_weights

    def get_edge_mask(self):
        return torch.sigmoid(self.edge_mask)

class CFFExplainer:
    """
    Implementation of CFF (Counterfactual Fairness for GNNs) method
    """
    def __init__(self, model, node_idx, edge_index, features, num_classes, 
                target=None, n_trials=100, alpha=0.5):
        self.model = model
        self.node_idx = node_idx
        # Get device from model
        self.device = next(model.parameters()).device
        self.edge_index = edge_index.clone().to(self.device)
        self.features = features.clone().to(self.device)
        self.num_classes = num_classes
        self.n_trials = n_trials  # Now used as number of training epochs
        self.alpha = alpha  # Balance between counterfactual and original predictions
        
        # Additional hyperparameters for the differentiable approach
        self.gamma = 1.0  # Margin parameter
        self.lambda_param = 1.0  # Weight for prediction loss
        self.lr = 0.01  # Learning rate
        
        with torch.no_grad():
            self.original_pred = model(self.features, self.edge_index)[self.node_idx].argmax().item()
        
        self.target = target if target is not None else 1 - self.original_pred
        
    def explain(self):
        # Initialize the explanation model
        explainer = ExplainModelNodeMulti(
            model=self.model,
            node_idx=self.node_idx,
            edge_index=self.edge_index,
            features=self.features
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.lr)
        
        # Best counterfactual tracking
        best_cf = None
        best_loss = float('inf')
        
        # Train the explainer
        for epoch in range(self.n_trials):
            explainer.train()
            optimizer.zero_grad()
            
            # Forward pass
            pred_cf, pred_orig = explainer()
            
            # Compute loss
            loss = explainer.loss(
                pred_cf[self.node_idx], 
                pred_orig[self.node_idx], 
                self.original_pred, 
                self.target, 
                self.gamma, 
                self.lambda_param, 
                self.alpha
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Check if we found a valid counterfactual
            with torch.no_grad():
                # Get edge mask
                edge_mask = explainer.get_edge_mask()
                
                # Create binary mask for edges to keep
                binary_mask = (edge_mask > 0.5)
                
                # Skip if no edges remain
                if not binary_mask.any():
                    continue
                
                # Apply mask to get counterfactual edge index
                cf_edge_index = self.edge_index[:, binary_mask]
                
                # Get prediction on counterfactual
                cf_logits = self.model(self.features, cf_edge_index)
                cf_pred = cf_logits[self.node_idx].argmax().item()
                
                # If counterfactual prediction matches target and loss improved
                if cf_pred == self.target and loss.item() < best_loss:
                    best_loss = loss.item()
                    
                    # Identify nodes connected in the counterfactual
                    connected_nodes = set()
                    for i in range(cf_edge_index.shape[1]):
                        connected_nodes.add(cf_edge_index[0, i].item())
                        connected_nodes.add(cf_edge_index[1, i].item())
                    
                    # Always include the target node
                    connected_nodes.add(self.node_idx)
                    
                    # Convert to list and sort for deterministic ordering
                    connected_nodes = sorted(list(connected_nodes))
                    
                    # Map to new indices
                    node_map = {old: new for new, old in enumerate(connected_nodes)}
                    
                    # Create new edge index and features
                    new_edge_index = torch.tensor([
                        [node_map[cf_edge_index[0, i].item()], node_map[cf_edge_index[1, i].item()]]
                        for i in range(cf_edge_index.shape[1])
                        if cf_edge_index[0, i].item() in node_map and cf_edge_index[1, i].item() in node_map
                    ], device=self.device).t()
                    
                    if new_edge_index.shape[1] == 0:  # No edges
                        new_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                    
                    new_features = self.features[connected_nodes]
                    new_node_idx = node_map[self.node_idx]
                    
                    best_cf = (new_edge_index, new_features, new_node_idx, torch.tensor(connected_nodes, device=self.device))
        
        # If no counterfactual found, return original graph
        if best_cf is None:
            return self.edge_index, self.features, self.node_idx, torch.arange(self.features.shape[0], device=self.device)
        
        return best_cf

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
        
        # Generate counterfactual explanation using CFF method
        cff_explainer = CFFExplainer(
            model=oracle_model,
            node_idx=subgraph_node_idx,
            edge_index=subgraph_data.edge_index,
            features=subgraph_data.x,
            num_classes=out,
            target=target_class,
            n_trials=100,
            alpha=0.5
        )
        
        cf_edge_index, cf_features, cf_node_idx, original_indices = cff_explainer.explain()
        
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
    results_df.to_csv(f'results/cff_{dataset_name}_results.csv', index=False)
    
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
    parser = argparse.ArgumentParser(description='CFF Explainer for counterfactual explanations')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset to use (Cora, Citeseer, AIDS, Facebook)')
    parser.add_argument('--samples', type=int, default=10, help='Number of test samples to explain')
    parser.add_argument('--hops', type=int, default=2, help='Number of hops for subgraph extraction')
    
    args = parser.parse_args()
    
    train_and_generate_cf_for_dataset(args.dataset, args.samples, args.hops)

if __name__ == "__main__":
    main() 