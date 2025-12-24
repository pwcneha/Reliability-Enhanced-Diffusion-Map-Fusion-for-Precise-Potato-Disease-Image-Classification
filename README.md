[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18045279.svg)](https://doi.org/10.5281/zenodo.18045279)

# Budget-Gated Fusion (BGF) — DMAP Reliability Gate

Research code accompanying the manuscript:

## 📄 Associated Publication

This repository contains the official implementation of the manuscript:

**Reliability-Enhanced Diffusion-Map Fusion for Precise Potato Disease Image Classification**  
(*Submitted to The Visual Computer*)

The code is permanently archived on Zenodo and can be cited using the following DOI:  
👉 https://doi.org/10.5281/zenodo.18045279

If you use this code, please cite the corresponding manuscript.

---

## What this repository does

Given:

- Diffusion Map (DMAP) features for train / validation / test splits (`.npz` files containing `X` and `y`)
- Calibrated baseline probabilities from a DMAP classifier (`.csv`)
- Calibrated expert probability outputs from multiple models (Snapshot Ensemble, Transformer, ViT, etc.)

the repository performs:

- Cluster-wise reliability estimation in DMAP space
- Budget-controlled expert overriding using expected negative log-likelihood (NLL)
- Reliability-aware probability fusion
- Quantitative evaluation including accuracy and NLL

The output consists of fused probability predictions and evaluation metrics.

---

## Installation

```bash
pip install -r requirements.txt
