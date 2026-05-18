from __future__ import annotations

import numpy as np
from sklearn.datasets import make_swiss_roll

from .config import ExperimentConfig


def labels_from_t(t: np.ndarray, reverse: bool = False) -> np.ndarray:
    cut_points = np.quantile(t, [1 / 3, 2 / 3])
    labels = np.digitize(t, cut_points)
    if reverse:
        labels = 2 - labels
    return labels.astype(int)


def sample_lift_basis(ambient_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gaussian_matrix = rng.normal(size=(ambient_dim, 3))
    q_matrix, _ = np.linalg.qr(gaussian_matrix)
    return q_matrix[:, :3].T


def lift_points(X_base: np.ndarray, ambient_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    basis = sample_lift_basis(ambient_dim, seed)
    X_lifted = X_base @ basis
    return X_lifted, basis


def make_standard_base(n_samples: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_base, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
    return X_base, labels_from_t(t), t


def make_holey_base(
    n_samples: int,
    noise: float,
    seed: int,
    hole_intervals: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch_size = max(int(n_samples * 2.5), n_samples + 400)
    X_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    collected = 0

    while collected < n_samples:
        X_batch, t_batch = make_swiss_roll(
            n_samples=batch_size,
            noise=noise,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        keep = np.ones(len(t_batch), dtype=bool)
        for low, high in hole_intervals:
            keep &= ~((t_batch >= low) & (t_batch <= high))
        X_parts.append(X_batch[keep])
        t_parts.append(t_batch[keep])
        collected += int(np.sum(keep))

    X_base = np.vstack(X_parts)[:n_samples]
    t = np.concatenate(t_parts)[:n_samples]
    return X_base, labels_from_t(t), t


def make_two_manifold_base(
    n_samples: int,
    noise: float,
    seed: int,
    offset: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_first = n_samples // 2
    n_second = n_samples - n_first

    X_first, t_first = make_swiss_roll(n_samples=n_first, noise=noise, random_state=seed)
    X_second, t_second = make_swiss_roll(n_samples=n_second, noise=noise, random_state=seed + 137)

    X_first = X_first - X_first.mean(axis=0)
    X_second = X_second - X_second.mean(axis=0)
    X_second[:, 0] *= -1
    X_second[:, 2] *= -1
    X_second += np.asarray(offset)

    y_first = labels_from_t(t_first, reverse=False)
    y_second = labels_from_t(t_second, reverse=True)

    X_base = np.vstack([X_first, X_second])
    y = np.concatenate([y_first, y_second])
    t = np.concatenate([t_first, t_second])

    order = np.random.default_rng(seed).permutation(len(y))
    return X_base[order], y[order], t[order]


def make_dataset(
    case_name: str,
    config: ExperimentConfig,
    noise: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if case_name == "standard":
        X_base, y, t = make_standard_base(config.n_samples, noise, seed)
    elif case_name == "holes":
        X_base, y, t = make_holey_base(config.n_samples, noise, seed, config.hole_intervals)
    elif case_name == "two_manifold":
        X_base, y, t = make_two_manifold_base(config.n_samples, noise, seed, config.two_manifold_offset)
    else:
        raise ValueError(f"Unknown case: {case_name}")

    X_lifted, basis = lift_points(X_base, config.ambient_dim, seed + 10_000)
    return {
        "X_base": X_base,
        "X_lifted": X_lifted,
        "y": y,
        "t": t,
        "basis": basis,
    }
