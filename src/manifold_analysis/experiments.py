from __future__ import annotations

import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .config import ExperimentConfig
from .data import make_dataset
from .methods import build_methods, fit_transform_full, fit_transform_split
from .metrics import neighbor_overlap_score


def run_experiments(config: ExperimentConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for case_name in config.case_order:
        for noise in config.noise_levels:
            for seed in config.seeds:
                dataset = make_dataset(case_name, config, noise, seed)
                X = dataset["X_lifted"]
                y = dataset["y"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=config.test_size,
                    stratify=y,
                    random_state=seed,
                )

                methods = build_methods(
                    embed_dim=config.embed_dim,
                    manifold_neighbors=config.manifold_neighbors,
                    seed=seed,
                )

                for method_name in config.method_order:
                    estimator = methods[method_name]
                    classifier = KNeighborsClassifier(n_neighbors=config.classifier_neighbors)
                    pipeline_start = time.perf_counter()

                    if method_name == "raw_knn":
                        classifier.fit(X_train, y_train)
                        accuracy = classifier.score(X_test, y_test)
                        neighbor_overlap = float("nan")
                        embed_time_sec = float("nan")
                    else:
                        Z_train, Z_test, embed_time_sec = fit_transform_split(estimator, X_train, X_test)
                        classifier.fit(Z_train, y_train)
                        accuracy = classifier.score(Z_test, y_test)

                        overlap_estimator = clone(estimator)
                        Z_full = fit_transform_full(overlap_estimator, X)
                        neighbor_overlap = neighbor_overlap_score(X, Z_full, k=config.overlap_neighbors)

                    pipeline_time_sec = time.perf_counter() - pipeline_start
                    rows.append(
                        {
                            "dataset_case": case_name,
                            "method": method_name,
                            "noise": noise,
                            "seed": seed,
                            "accuracy": accuracy,
                            "neighbor_overlap": neighbor_overlap,
                            "embed_time_sec": embed_time_sec,
                            "pipeline_time_sec": pipeline_time_sec,
                            "ambient_dim": X.shape[1],
                            "embed_dim": config.embed_dim,
                        }
                    )

    return prepare_results_dataframe(pd.DataFrame(rows), config)


def prepare_results_dataframe(results_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    results = results_df.copy()
    results["dataset_case"] = pd.Categorical(results["dataset_case"], categories=config.case_order, ordered=True)
    results["method"] = pd.Categorical(results["method"], categories=config.method_order, ordered=True)
    results["accuracy_pct"] = 100 * results["accuracy"]
    results["neighbor_overlap_pct"] = 100 * results["neighbor_overlap"]
    results["embed_time_ms"] = 1000 * results["embed_time_sec"]
    results["pipeline_time_ms"] = 1000 * results["pipeline_time_sec"]
    return results.sort_values(["dataset_case", "method", "noise", "seed"]).reset_index(drop=True)


def summarize_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_df = (
        results_df.groupby(["dataset_case", "method", "noise"], observed=True)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            neighbor_overlap_mean=("neighbor_overlap", "mean"),
            neighbor_overlap_std=("neighbor_overlap", "std"),
            embed_time_mean=("embed_time_sec", "mean"),
            embed_time_std=("embed_time_sec", "std"),
            pipeline_time_mean=("pipeline_time_sec", "mean"),
            pipeline_time_std=("pipeline_time_sec", "std"),
        )
        .reset_index()
    )

    case_method_summary_df = (
        results_df.groupby(["dataset_case", "method"], observed=True)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            neighbor_overlap_mean=("neighbor_overlap", "mean"),
            embed_time_mean=("embed_time_sec", "mean"),
            pipeline_time_mean=("pipeline_time_sec", "mean"),
        )
        .reset_index()
    )

    return summary_df, case_method_summary_df
