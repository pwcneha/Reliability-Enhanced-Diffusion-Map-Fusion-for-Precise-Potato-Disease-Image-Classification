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



