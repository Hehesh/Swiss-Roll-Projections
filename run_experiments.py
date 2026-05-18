from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from scipy.sparse import SparseEfficiencyWarning

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl-cache"))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manifold_analysis import ExperimentConfig, run_experiments, summarize_results
from manifold_analysis.plotting import (
    configure_matplotlib,
    ensure_output_dirs,
    plot_accuracy_vs_noise,
    plot_dataset_cases,
    plot_embedding_gallery,
    plot_neighbor_overlap,
    plot_runtime_tradeoff,
    plot_standard_geometry,
    save_dataset_views,
    save_embedding_views,
    save_figure,
)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The number of connected components of the neighbors graph is",
    )
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    configure_matplotlib()

    config = ExperimentConfig(project_root=PROJECT_ROOT)
    ensure_output_dirs(config)

    print(f"Running experiments in ambient dimension R^{config.ambient_dim}")
    print(f"Dataset cases: {list(config.case_order)}")
    print(f"Methods: {list(config.method_order)}")

    results_df = run_experiments(config)
    summary_df, case_method_summary_df = summarize_results(results_df)

    results_df.to_csv(config.tables_dir / "results_raw.csv", index=False)
    summary_df.to_csv(config.tables_dir / "results_summary.csv", index=False)
    case_method_summary_df.to_csv(config.tables_dir / "results_case_summary.csv", index=False)

    save_figure(plot_standard_geometry(config), config.figures_dir / "swiss_roll_3d", formats=("png",))
    save_figure(plot_dataset_cases(config), config.figures_dir / "extension_cases_3d", formats=("png",))
    save_dataset_views(config)
    save_figure(plot_embedding_gallery(config), config.figures_dir / "embedding_gallery_clean")
    save_embedding_views(config)
    save_figure(plot_accuracy_vs_noise(summary_df, config), config.figures_dir / "accuracy_vs_noise")
    save_figure(plot_neighbor_overlap(summary_df, config), config.figures_dir / "neighbor_overlap_vs_noise")
    save_figure(plot_runtime_tradeoff(results_df, config), config.figures_dir / "runtime_comparison")

    print(f"Saved tables to {config.tables_dir}")
    print(f"Saved figures to {config.figures_dir}")


if __name__ == "__main__":
    main()
