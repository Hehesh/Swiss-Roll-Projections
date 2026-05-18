from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CASE_ORDER = ("standard", "holes", "two_manifold")
CASE_LABELS = {
    "standard": "Standard Swiss Roll",
    "holes": "Swiss Roll with Holes",
    "two_manifold": "Two-Manifold Swiss Roll",
}
METHOD_ORDER = ("raw_knn", "pca", "gaussian_rp", "sparse_rp", "isomap", "lle")
METHOD_LABELS = {
    "raw_knn": "Raw k-NN",
    "pca": "PCA",
    "gaussian_rp": "Gaussian RP",
    "sparse_rp": "Sparse RP",
    "isomap": "Isomap",
    "lle": "LLE",
}
METHOD_SHORT = {
    "raw_knn": "Raw",
    "pca": "PCA",
    "gaussian_rp": "GRP",
    "sparse_rp": "SRP",
    "isomap": "Iso",
    "lle": "LLE",
}
METHOD_COLORS = {
    "raw_knn": "#4c72b0",
    "pca": "#dd8452",
    "gaussian_rp": "#55a868",
    "sparse_rp": "#c44e52",
    "isomap": "#8172b3",
    "lle": "#937860",
}
VIEW_SETTINGS = {
    "standard": (14, -66),
    "holes": (14, -63),
    "two_manifold": (18, -52),
}


@dataclass(frozen=True)
class ExperimentConfig:
    case_order: tuple[str, ...] = CASE_ORDER
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35)
    seeds: tuple[int, ...] = (0, 1, 2)
    method_order: tuple[str, ...] = METHOD_ORDER
    n_samples: int = 1200
    ambient_dim: int = 30
    embed_dim: int = 2
    manifold_neighbors: int = 12
    classifier_neighbors: int = 7
    overlap_neighbors: int = 10
    test_size: float = 0.25
    hole_intervals: tuple[tuple[float, float], ...] = ((8.0, 9.25), (10.75, 12.0))
    two_manifold_offset: tuple[float, float, float] = (4.0, 2.5, -3.5)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def figures_dir(self) -> Path:
        return self.results_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def dataset_figures_dir(self) -> Path:
        return self.figures_dir / "datasets"

    @property
    def embedding_figures_dir(self) -> Path:
        return self.figures_dir / "embeddings"
