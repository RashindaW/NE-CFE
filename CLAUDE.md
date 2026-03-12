# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NE-CFE (Node and Edge Counterfactual Explanations) — a graph counterfactual explanation method that generates minimal perturbations to node features and edges to flip a GNN classifier's prediction. The repository also includes implementations of SOTA comparison methods (EGO, CF-GNN, CFF, Random).

## Running Commands

### NE-CFE (main algorithm)
```bash
python NE_CFE.py --dataset citeseer --model-type gcn --alpha-policy linear --k-hop 2 --epochs 500
```

Key flags: `--dataset`, `--model-type` (gcn/chebnet), `--alpha-policy`, `--cuda N`, `--force-cpu`, `--nodes N`, `--visualize`, `--load-model PATH`, `--dataset-path PATH` (for Cornell fallback).

### SOTA comparison methods
```bash
cd SOTA_comparison
python counterfactual_evaluation.py --dataset Cora --methods ego cf_gnn cff random --samples 100 --model gcn --cuda 0
```

Note: dataset names are capitalized in the SOTA comparison script (`Citeseer`, `Cora`, `Cornell`).

### Load and visualize saved results
```bash
python SOTA_comparison/load_results.py path/to/results.pt
```

## Architecture

### NE_CFE.py — Single-file end-to-end pipeline

The entire NE-CFE algorithm lives in one file with this flow:

1. **Dataset loading** — Supports Planetoid (Cora/CiteSeer), WebKB (Cornell), FacebookPagePage, and TUDataset (AIDS). Each dataset has different loading logic, mask creation, and feature type handling.
2. **Oracle GNN training** — Trains a GCN or ChebNet classifier on the full graph. Can load pretrained models via `--load-model`.
3. **k-hop subgraph extraction** — For each test node, extracts a local neighborhood subgraph.
4. **Counterfactual optimization** — The `GraphPerturber` (nn.Module) jointly optimizes:
   - **Node feature perturbations** (`P_x` parameter): continuous deltas clamped to valid ranges; binary features use Binary-Concrete relaxation with temperature annealing
   - **Edge mask** (`EP_x` parameter): soft mask over edges, thresholded to produce discrete edge additions/removals
   - **Alpha scheduling**: trades off node-feature sparsity vs edge sparsity over epochs
5. **Evaluation & saving** — Computes success rates, feature/edge change counts, distances (L1, L2, cosine, HEOM), and saves results to `results/`.

### Key technical details

- Features are classified as **discrete** (all values are 0 or 1) or **continuous** per-dimension using `DISCRETE_MASK`/`CONTINUOUS_MASK`
- HEOM (Heterogeneous Euclidean-Overlap Metric) handles mixed feature types: normalized L2 for continuous, overlap for discrete
- Perturbations are **added** to base features (not overwritten), then clamped to valid ranges
- Edge handling preserves undirectedness and self-loops
- The GCN/ChebNet model classes are duplicated across files (NE_CFE.py and each SOTA file). They must stay consistent.

### SOTA_comparison/ — Baseline methods

Each file implements one explainer class with a common interface:
- `ego_counterfactual_trials.py` → `EGOExplainer` (feature-only perturbation)
- `cf_gnn_counterfactual_trials.py` → `CFGNNExplainer` (edge-only via CE loss)
- `cff_counterfactual_trials.py` → `CFFExplainer` (edge-only via margin loss)
- `random_counterfactual_trials.py` → `RandomExplainer` (random edge+feature perturbation)

`counterfactual_evaluation.py` orchestrates running all methods, shares the same GCN/ChebNet model definitions, and produces CSV results and `.pt` result files.

## Dependencies

PyTorch, PyTorch Geometric (`torch_geometric`), NumPy, pandas, matplotlib, tqdm, scikit-learn.

Install: `pip install -r requirements.txt` (PyG requires separate installation per CUDA version).
