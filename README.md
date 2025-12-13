\# Budget-Gated Fusion (BGF) — DMAP Reliability Gate



Research code accompanying the manuscript:

\*\*"Budget-Gated Reliability-Aware Diffusion-Map Ensemble for Potato Disease Image Classification"\*\*

(Submitted to The Visual Computer)



\## What this repository does

Given:

\- DMAP features for train/val/test (`.npz` with `X` and `y`)

\- Base (DMAP) calibrated probabilities for val/test (`.csv`)

\- Expert probability CSVs for val/test (Snapshot / Transformer / ViT, etc.)



It performs cluster-wise reliability gating with a p-budget and outputs fused probabilities + metrics.



\## Install

```bash

pip install -r requirements.txt
## Quick Start (Example)
```md
## Input Data Format

This repository assumes that feature extraction and model training have already been performed.
The required inputs are:

### DMAP Feature Files (`.npz`)
Each `.npz` file must contain:
- `X`: a NumPy array of shape `(N, d)` representing diffusion-map embeddings
- `y`: a NumPy array of shape `(N,)` containing integer class labels

### Probability Files (`.csv`)
Each `.csv` file must contain class probabilities of shape `(N, C)`:
- One row per sample
- One column per class
- Each row must sum to 1

Baseline probabilities correspond to the DMAP classifier, while expert probabilities correspond to individual models (e.g., Snapshot Ensemble, Transformer, ViT).

After installing the dependencies, the fusion pipeline can be executed as follows:

```bash
python run_fusion.py \
  --dmap_train path/to/train_dmap.npz \
  --dmap_val   path/to/val_dmap.npz \
  --dmap_test  path/to/test_dmap.npz \
  --base_val   path/to/dmap_probs_val.csv \
  --base_test  path/to/dmap_probs_test.csv \
  --expert snapshot    path/to/snapshot_probs_val.csv    path/to/snapshot_probs_test.csv \
  --expert transformer path/to/transformer_probs_val.csv path/to/transformer_probs_test.csv \
  --expert vit         path/to/vit_probs_val.csv         path/to/vit_probs_test.csv \
  --out_dir outputs/run1





