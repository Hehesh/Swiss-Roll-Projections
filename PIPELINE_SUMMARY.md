# Pipeline Summary

The project notebook is now a self-contained report at [manifold_project.ipynb](/Users/braydenchien/Desktop/MAT%20126/Project/manifold_project.ipynb).

## What Changed

- The Swiss roll is no longer analyzed only in 3D.
- Each dataset is first generated in coordinates `(x1, x2, x3)`.
- The data is then lifted into `R^d` by mapping each point to `x1 v1 + x2 v2 + x3 v3`, where `v1`, `v2`, and `v3` are vectors in the higher-dimensional ambient space.
- Results are displayed inline in the notebook instead of being saved out as external CSVs or figure files.
- A new time-vs-accuracy analysis has been added.

## Dataset Cases

- `standard`: a single Swiss roll with labels from tertiles of the intrinsic roll coordinate
- `holes`: the same Swiss roll with missing intervals removed from the roll coordinate
- `two_manifold`: two nearby Swiss rolls with reversed label ordering on the second roll

## Methods Compared

- raw `k`-NN on the lifted ambient coordinates
- PCA + `k`-NN
- Gaussian random projection + `k`-NN
- sparse random projection + `k`-NN
- Isomap + `k`-NN
- LLE + `k`-NN

## Metrics and Visuals

- classification accuracy
- neighborhood preservation
- embedding runtime
- total pipeline runtime
- accuracy vs noise
- neighborhood preservation vs noise
- time vs accuracy

## Notebook Structure

The notebook now includes:

- a written problem description
- a high-level description of the solution
- a presentation-feedback section with code changes
- the data-loading and algorithm code
- inline tables and figures
- a final discussion of what was learned from the project
