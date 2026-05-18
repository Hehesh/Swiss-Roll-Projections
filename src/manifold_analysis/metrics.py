from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def neighbor_overlap_score(X_ref: np.ndarray, X_emb: np.ndarray, k: int) -> float:
    ref_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X_ref).kneighbors(return_distance=False)[:, 1:]
    emb_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X_emb).kneighbors(return_distance=False)[:, 1:]
    overlaps = []
    for ref_idx, emb_idx in zip(ref_neighbors, emb_neighbors):
        overlaps.append(len(set(ref_idx).intersection(set(emb_idx))) / k)
    return float(np.mean(overlaps))
