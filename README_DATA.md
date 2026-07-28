# Data availability and repository scope

## IARI cohort

The primary evaluation used 866 potato-leaf images from the IARI cohort:

- Early Blight: 225 images
- Healthy: 347 images
- Late Blight: 294 images
- Acquisition grouping units: 836

The raw IARI images are not redistributed in this software repository. Access
requests should follow the data-availability statement in the associated
manuscript and remain subject to institutional permission.

## PlantVillage benchmark

The separately reconstructed PlantVillage benchmark is also not redistributed
here. It was assembled from the public source described in the manuscript.
Users wishing to reproduce that benchmark should obtain the source images
independently and apply the documented reconstruction and partition rules.

## Files supplied in this repository

The repository provides the material that can be shared without redistributing
the raw image collections:

- the as-executed and equation-consistent BGF implementations;
- the locked BGF configuration;
- regional-policy and intervention-count audit files;
- runtime input-schema documentation; and
- scripts for constructing fold-local prediction locks without using held-out
  labels.

## User-supplied runtime input

Running `run_fusion.py` requires a fold-local NPZ file containing the
development representations, development labels, held-out representations,
and calibrated baseline and expert probabilities described in `README.md`.
Held-out labels must not be included in this construction input.

No Google Drive path, credential, private image, or institutional identifier is
required by the released source code.
