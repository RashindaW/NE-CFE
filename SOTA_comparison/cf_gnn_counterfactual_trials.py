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

def k_hop_graph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None):
    # Extract the k-hop subgraph of node_idx from edge_index
    device = edge_index.device
    
    num_nodes = edge_index.max().item() + 1 if num_nodes is None else num_nodes
    
    # Convert to CPU for subgraph extraction
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
    
    return subset, sub_edge_index, mapping

class CFGNNExplainer:
    """
    Implementation of the CF-GNN method for counterfactual explanations
    """
    def __init__(self, model, node_idx, edge_index, features, target=None, epochs=100, num_classes=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.node_idx = node_idx
        self.edge_index = edge_index.to(self.device)
        self.features = features.to(self.device)
        self.epochs = epochs
        self.lr = 0.01
        self.num_classes = num_classes  # This is optional in our implementation
        self.hops = 2  # Default value
        
        with torch.no_grad():
            self.original_pred = model(self.features, self.edge_index)[self.node_idx].argmax().item()
        
        self.target = target if target is not None else 1 - self.original_pred
    
    def explain(self):
        # Extract k-hop subgraph around the node
        subset, sub_edge_index, mapping = k_hop_graph(
            node_idx=self.node_idx,
            num_hops=self.hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.features.shape[0]
        )
        
        # Get the index of the target node in the subgraph
        sub_node_idx = mapping.item()
        
        # Extract features for the subgraph - UNCHANGED as we only modify edges
        sub_features = self.features[subset].clone()
        
        # Check if the prediction is already the target class
        with torch.no_grad():
            sub_pred = self.model(sub_features, sub_edge_index)[sub_node_idx].argmax().item()
        
        if sub_pred == self.target:
            return sub_edge_index, sub_features
        
        # Create edge mask for optimization - one value per edge
        num_edges = sub_edge_index.size(1)
        edge_mask = torch.ones(num_edges, device=self.device, requires_grad=True)
        
        # Define optimizer for the edge mask
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)
        
        # Optimization loop
        best_cf_found = False
        best_edge_index = sub_edge_index.clone()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Apply sigmoid to edge mask to bound values between 0 and 1
            sigmoid_mask = torch.sigmoid(edge_mask)
            
            # To use the edge mask in message passing, we need to create a sparse representation
            # For each edge, if mask value > 0.5, keep the edge, otherwise drop it
            # During training, we use a soft mask for gradient flow
            edge_weights = sigmoid_mask
            
            # Get model prediction with weighted edges
            logits = self.model(sub_features, sub_edge_index, edge_weight=edge_weights)
            pred_logits = logits[sub_node_idx]
            
            # Calculate loss to encourage target class prediction
            target_tensor = torch.tensor([self.target], device=self.device)
            ce_loss = F.cross_entropy(pred_logits.unsqueeze(0), target_tensor)
            
            # L1 regularization to encourage sparsity in the mask (prefer edge deletion)
            mask_l1_loss = 0.1 * torch.sum(sigmoid_mask)
            
            # Total loss
            loss = ce_loss + mask_l1_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Check if we've found a valid counterfactual
            # For evaluation, apply hard threshold on the mask
            with torch.no_grad():
                hard_mask = (sigmoid_mask > 0.5)
                if not hard_mask.any():  # Ensure we have at least one edge
                    continue
                    
                # Create CF edge index by selecting edges where mask is 1
                cf_edge_index = sub_edge_index[:, hard_mask]
                
                # Check if counterfactual exists and prediction changes
                if cf_edge_index.size(1) > 0:  # Make sure we have edges
                    cf_pred = self.model(sub_features, cf_edge_index)[sub_node_idx].argmax().item()
                    if cf_pred == self.target:
                        best_edge_index = cf_edge_index.clone()
                        best_cf_found = True
                        break
        
        # If no valid counterfactual found, apply final threshold and return best attempt
        if not best_cf_found:
            with torch.no_grad():
                hard_mask = (torch.sigmoid(edge_mask) > 0.5)
                if hard_mask.any():  # Ensure we have at least one edge
                    best_edge_index = sub_edge_index[:, hard_mask]
        
        # Return the counterfactual subgraph with UNCHANGED features
        return best_edge_index, sub_features

def load_data(dataset_name):
    if dataset_name.lower() == 'cora':
        dataset = Planetoid(root='./data/Cora', name='Cora')
    elif dataset_name.lower() == 'citeseer':
        dataset = Planetoid(root='./data/Citeseer', name='Citeseer')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    data = dataset[0]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    
    return data

def train_oracle(model, g, epochs=400, lr=1e-2, wd=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(g.x, g.edge_index)
        loss = F.nll_loss(out[g.train_mask], g.y[g.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(g.x, g.edge_index)
        acc = (logits.argmax(1)[g.test_mask] == g.y[g.test_mask]).float().mean().item()

    return acc

def train_and_generate_cf_for_dataset(dataset_name, num_samples=10, hops=2):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(dataset_name)
    
    # Set up model
    model = GCN(in_channels=data.x.size(1), hidden_channels=16, out_channels=data.y.max().item()+1).to(device)
    
    # Train model
    accuracy = train_oracle(model, data)
    print(f"Oracle accuracy: {accuracy:.4f}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    
    # Get correctly classified nodes of each class
    correct_nodes = []
    for c in range(data.y.max().item() + 1):
        class_nodes = (data.y == c).nonzero(as_tuple=True)[0]
        correct_class_nodes = class_nodes[(pred[class_nodes] == c)]
        if len(correct_class_nodes) > 0:
            correct_nodes.append(correct_class_nodes)
    
    # Sample nodes for counterfactual generation
    cf_results = []
    
    for i, class_nodes in enumerate(correct_nodes):
        # Set target as the opposite class (for binary) or random other class
        if len(correct_nodes) == 2:  # Binary case
            target = 1 - i
        else:  # Multi-class
            all_classes = list(range(len(correct_nodes)))
            all_classes.remove(i)
            target = np.random.choice(all_classes)
        
        # Sample nodes from this class
        if len(class_nodes) > num_samples:
            sampled_nodes = class_nodes[torch.randperm(len(class_nodes))[:num_samples]]
        else:
            sampled_nodes = class_nodes
        
        print(f"Processing {len(sampled_nodes)} nodes from class {i} with target {target}")
        
        for node_idx in sampled_nodes:
            node_idx = node_idx.item()
            
            # Generate counterfactual
            cf_explainer = CFGNNExplainer(
                model=model,
                node_idx=node_idx,
                edge_index=data.edge_index,
                features=data.x,
                target=target
            )
            
            start_time = time.time()
            cf_edge_index, cf_features = cf_explainer.explain()
            end_time = time.time()
            
            # Check if the counterfactual works
            with torch.no_grad():
                original_prob = model(data.x, data.edge_index)[node_idx, i].exp().item()
                cf_prob = model(cf_features, cf_edge_index)[node_idx, target].exp().item()
                cf_pred = model(cf_features, cf_edge_index)[node_idx].argmax().item()
            
            # Calculate edge changes (no more feature changes)
            success = cf_pred == target
            
            result = {
                'node_idx': node_idx,
                'original_class': i,
                'target_class': target,
                'success': success,
                'original_prob': original_prob,
                'cf_prob': cf_prob,
                'time': end_time - start_time
            }
            
            cf_results.append(result)
            
            print(f"Node {node_idx}: {'Success' if success else 'Failed'} - {cf_prob:.4f} for target class {target}")
    
    # Compute overall metrics
    results_df = pd.DataFrame(cf_results)
    
    success_rate = results_df['success'].mean()
    avg_time = results_df['time'].mean()
    
    print(f"\nResults for {dataset_name}:")
    print(f"Success rate: {success_rate:.4f}")
    print(f"Average time: {avg_time:.4f} seconds")
    
    return results_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name (cora or citeseer)')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples per class')
    parser.add_argument('--hops', type=int, default=2, help='Number of hops for subgraph extraction')
    args = parser.parse_args()
    
    print(f"Running CF-GNN on {args.dataset} with {args.samples} samples per class")
    results = train_and_generate_cf_for_dataset(args.dataset, args.samples, args.hops)

if __name__ == "__main__":
    main() 