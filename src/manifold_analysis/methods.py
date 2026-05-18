from __future__ import annotations

import time

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


def to_dense(array_like) -> np.ndarray:
    if issparse(array_like):
        return array_like.toarray()
    return np.asarray(array_like)


def build_methods(embed_dim: int, manifold_neighbors: int, seed: int) -> dict[str, object | None]:
    return {
        "raw_knn": None,
        "pca": PCA(n_components=embed_dim, random_state=seed),
        "gaussian_rp": GaussianRandomProjection(n_components=embed_dim, random_state=seed),
        "sparse_rp": SparseRandomProjection(n_components=embed_dim, random_state=seed),
        "isomap": Isomap(n_components=embed_dim, n_neighbors=manifold_neighbors, n_jobs=1),
        "lle": LocallyLinearEmbedding(
            n_components=embed_dim,
            n_neighbors=manifold_neighbors,
            method="standard",
            eigen_solver="arpack",
        ),
    }


def fit_transform_split(estimator, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    if hasattr(estimator, "fit_transform"):
        Z_train = estimator.fit_transform(X_train)
    else:
        estimator.fit(X_train)
        Z_train = estimator.transform(X_train)
    Z_test = estimator.transform(X_test)
    elapsed = time.perf_counter() - start
    return to_dense(Z_train), to_dense(Z_test), elapsed


def fit_transform_full(estimator, X: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "fit_transform"):
        Z = estimator.fit_transform(X)
    else:
        estimator.fit(X)
        Z = estimator.transform(X)
    return to_dense(Z)
