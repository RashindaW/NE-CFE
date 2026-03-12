# NE-CFE: Node and Edge Counterfactual Explanations for Graph Neural Networks

NE-CFE generates minimal counterfactual explanations for node-level GNN predictions by jointly optimizing perturbations to both **node features** and **graph edges**. Given a trained GNN classifier and a target node, NE-CFE finds the smallest changes needed to flip the classifier's prediction.

## Key Features

- **Joint node-feature and edge perturbation** — unlike methods that only modify edges (CF-GNN, CFF) or only features (EGO), NE-CFE optimizes both simultaneously
- **Binary-Concrete relaxation** — uses Gumbel-Softmax for differentiable optimization of discrete (binary) features with temperature annealing
- **HEOM distance metric** — properly handles mixed continuous/discrete feature spaces via the Heterogeneous Euclidean-Overlap Metric
- **Alpha scheduling** — configurable policies (`linear`, `cosine`, `exponential`, etc.) to trade off node-feature sparsity vs edge sparsity during optimization
- **Multiple GNN architectures** — supports GCN and ChebNet as oracle classifiers
- **Multiple datasets** — CiteSeer, Cora, Facebook, AIDS, Cornell

## Installation

```bash
pip install -r requirements.txt
```

PyTorch Geometric requires separate installation depending on your CUDA version. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

### Running NE-CFE

```bash
python NE_CFE.py --dataset citeseer --model-type gcn --alpha-policy linear --k-hop 2 --epochs 500
```

#### Key Arguments

| Argument | Options | Default | Description |
|---|---|---|---|
| `--dataset` | `citeseer`, `cora`, `facebook`, `aids`, `cornell` | `citeseer` | Dataset to use |
| `--model-type` | `gcn`, `chebnet` | `gcn` | Oracle GNN architecture |
| `--alpha-policy` | `linear`, `constant`, `inverse`, `cosine`, `exponential`, `sinusoidal`, `dynamic` | `linear` | How to balance node vs edge sparsity over epochs |
| `--k-hop` | integer | `2` | Neighborhood subgraph radius |
| `--epochs` | integer | `500` | CF optimization epochs |
| `--lr` | float | `0.002` | CF optimization learning rate |
| `--cuda` | integer | `0` | CUDA device index |
| `--force-cpu` | flag | — | Force CPU execution |
| `--nodes` | integer | all | Limit number of test nodes to explain |
| `--visualize` | flag | — | Save result visualizations |
| `--load-model` | path | — | Load a pretrained oracle instead of training |
| `--dataset-path` | path | — | Custom dataset path (required for Cornell if WebKB download fails) |

### Running SOTA Baselines

Compare against EGO, CF-GNN, CFF, and Random baselines:

```bash
cd SOTA_comparison
python counterfactual_evaluation.py --dataset Cora --methods ego cf_gnn cff random --samples 100 --model gcn --cuda 0
```

> **Note:** Dataset names are capitalized in the SOTA comparison script (`Cora`, `Citeseer`, `Cornell`).

### Visualizing Results

```bash
cd SOTA_comparison
python load_results.py path/to/results.pt
```

## Repository Structure

```
├── NE_CFE.py                          # Main NE-CFE algorithm (end-to-end pipeline)
├── SOTA_comparison/
│   ├── counterfactual_evaluation.py   # Unified evaluation harness for baselines
│   ├── ego_counterfactual_trials.py   # EGO explainer (feature-only perturbation)
│   ├── cff_counterfactual_trials.py   # CFF explainer (edge-mask optimization)
│   ├── cf_gnn_counterfactual_trials.py # CF-GNN explainer (edge perturbation via CE loss)
│   ├── random_counterfactual_trials.py # Random baseline
│   └── load_results.py               # Result loading and visualization
├── requirements.txt
└── README.md
```

## How It Works

1. **Train an oracle GNN** on the full graph (GCN or ChebNet)
2. **Extract k-hop subgraphs** around each test node
3. **Optimize counterfactual perturbations** via the `GraphPerturber` module:
   - Learns continuous feature deltas (clamped to valid ranges)
   - Uses Binary-Concrete for discrete features
   - Learns a soft edge mask (sigmoid-gated)
   - Loss = cross-entropy (flip prediction) + α · node sparsity + (1-α) · edge sparsity + HEOM distance
4. **Threshold and evaluate** — hard-threshold the soft perturbations and compute metrics (fidelity, sparsity, distances)

## Output Metrics

- **Success rate** — fraction of test nodes where the prediction was flipped
- **Feature/edge sparsity** — how many features/edges were changed (absolute and relative)
- **Fidelity (Ψ)** — whether the oracle was correct on the original and the CF disagrees
- **Distance metrics** — L1, L2, cosine, Hamming, HEOM between original and counterfactual features
- **Embedding distance** — L2 distance between graph embeddings of original and CF subgraphs
