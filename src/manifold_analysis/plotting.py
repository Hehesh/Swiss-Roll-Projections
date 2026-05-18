from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.base import clone

from .config import (
    CASE_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_SHORT,
    VIEW_SETTINGS,
    ExperimentConfig,
)
from .data import make_dataset
from .methods import build_methods, fit_transform_full


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")


def ensure_output_dirs(config: ExperimentConfig) -> None:
    for path in (
        config.figures_dir,
        config.dataset_figures_dir,
        config.embedding_figures_dir,
        config.tables_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, destination: Path, formats: tuple[str, ...] = ("png", "svg")) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        fig.savefig(destination.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_standard_geometry(config: ExperimentConfig) -> plt.Figure:
    dataset = make_dataset("standard", config, noise=0.0, seed=0)
    X_base = dataset["X_base"]
    y = dataset["y"]

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    scatter = ax.scatter(X_base[:, 0], X_base[:, 1], X_base[:, 2], c=y, cmap="viridis", s=10, alpha=0.82)
    ax.view_init(*VIEW_SETTINGS["standard"])
    ax.set_title(CASE_LABELS["standard"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.colorbar(scatter, ax=ax, shrink=0.72, label="Class")
    return fig


def plot_dataset_cases(config: ExperimentConfig) -> plt.Figure:
    fig = plt.figure(figsize=(18, 5.5))
    scatter = None

    for idx, case_name in enumerate(config.case_order, start=1):
        dataset = make_dataset(case_name, config, noise=0.0, seed=0)
        X_base = dataset["X_base"]
        y = dataset["y"]

        ax = fig.add_subplot(1, len(config.case_order), idx, projection="3d")
        scatter = ax.scatter(X_base[:, 0], X_base[:, 1], X_base[:, 2], c=y, cmap="viridis", s=10, alpha=0.82)
        ax.view_init(*VIEW_SETTINGS[case_name])
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig.suptitle(f"Underlying 3D Geometry Before Lifting to R^{config.ambient_dim}", fontsize=16)
    if scatter is not None:
        fig.colorbar(scatter, ax=fig.axes, shrink=0.72, label="Class")
    return fig


def save_dataset_views(config: ExperimentConfig) -> None:
    for case_name in config.case_order:
        dataset = make_dataset(case_name, config, noise=0.0, seed=0)
        X_base = dataset["X_base"]
        y = dataset["y"]

        fig = plt.figure(figsize=(7.2, 5.6))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        scatter = ax.scatter(X_base[:, 0], X_base[:, 1], X_base[:, 2], c=y, cmap="viridis", s=10, alpha=0.82)
        ax.view_init(*VIEW_SETTINGS[case_name])
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(scatter, ax=ax, shrink=0.72, label="Class")
        save_figure(fig, config.dataset_figures_dir / f"{case_name}_swiss_roll")


def plot_embedding_gallery(config: ExperimentConfig) -> plt.Figure:
    clean_dataset = make_dataset("standard", config, noise=0.0, seed=0)
    gallery_methods = build_methods(config.embed_dim, config.manifold_neighbors, seed=0)
    gallery_names = ["pca", "gaussian_rp", "sparse_rp", "isomap", "lle"]

    fig, axes = plt.subplots(1, len(gallery_names), figsize=(24, 4.6), constrained_layout=True)
    for ax, method_name in zip(axes, gallery_names):
        embedding = fit_transform_full(clone(gallery_methods[method_name]), clean_dataset["X_lifted"])
        ax.scatter(embedding[:, 0], embedding[:, 1], c=clean_dataset["y"], cmap="viridis", s=10, alpha=0.8)
        ax.set_title(METHOD_LABELS[method_name])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)

    fig.suptitle("Clean Standard Case After Lifting: 2D Embedding Comparison", fontsize=16)
    return fig


def save_embedding_views(config: ExperimentConfig) -> None:
    clean_dataset = make_dataset("standard", config, noise=0.0, seed=0)
    methods = build_methods(config.embed_dim, config.manifold_neighbors, seed=0)

    for method_name in ("pca", "gaussian_rp", "sparse_rp", "isomap", "lle"):
        embedding = fit_transform_full(clone(methods[method_name]), clean_dataset["X_lifted"])
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.scatter(embedding[:, 0], embedding[:, 1], c=clean_dataset["y"], cmap="viridis", s=10, alpha=0.8)
        ax.set_title(METHOD_LABELS[method_name])
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(False)
        save_figure(fig, config.embedding_figures_dir / f"clean_{method_name}")


def plot_accuracy_vs_noise(summary_df: pd.DataFrame, config: ExperimentConfig) -> plt.Figure:
    fig, axes = plt.subplots(1, len(config.case_order), figsize=(18, 5.4), sharey=True)
    fig.subplots_adjust(bottom=0.34, top=0.83, wspace=0.18)

    for ax, case_name in zip(axes, config.case_order):
        case_df = summary_df[summary_df["dataset_case"] == case_name]
        for method_name in config.method_order:
            method_df = case_df[case_df["method"] == method_name].sort_values("noise")
            ax.errorbar(
                method_df["noise"],
                method_df["accuracy_mean"],
                yerr=method_df["accuracy_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                label=METHOD_LABELS[method_name],
            )
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("Noise level $\\sigma$")
        ax.set_ylim(0.25, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(False)

    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle("Accuracy vs Noise on Lifted Swiss Roll Data", fontsize=16)
    return fig


def plot_neighbor_overlap(summary_df: pd.DataFrame, config: ExperimentConfig) -> plt.Figure:
    embed_methods = [name for name in config.method_order if name != "raw_knn"]
    fig, axes = plt.subplots(1, len(config.case_order), figsize=(18, 5.4), sharey=True)
    fig.subplots_adjust(bottom=0.34, top=0.83, wspace=0.18)

    for ax, case_name in zip(axes, config.case_order):
        case_df = summary_df[summary_df["dataset_case"] == case_name]
        for method_name in embed_methods:
            method_df = case_df[case_df["method"] == method_name].sort_values("noise")
            ax.errorbar(
                method_df["noise"],
                method_df["neighbor_overlap_mean"],
                yerr=method_df["neighbor_overlap_std"].fillna(0.0),
                marker="o",
                linewidth=2,
                capsize=3,
                label=METHOD_LABELS[method_name],
            )
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("Noise level $\\sigma$")
        ax.set_ylim(0.0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(False)

    axes[0].set_ylabel(f"Shared {config.overlap_neighbors}-NN fraction (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle("Neighborhood Preservation vs Noise on Lifted Data", fontsize=16)
    return fig


def plot_runtime_tradeoff(results_df: pd.DataFrame, config: ExperimentConfig) -> plt.Figure:
    time_accuracy_df = (
        results_df.groupby(["dataset_case", "method"], observed=True)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            pipeline_time_mean=("pipeline_time_ms", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(config.case_order), figsize=(18, 5), sharey=True)
    fig.subplots_adjust(bottom=0.22, top=0.83, wspace=0.18)

    for ax, case_name in zip(axes, config.case_order):
        case_df = time_accuracy_df[time_accuracy_df["dataset_case"] == case_name].copy()
        ax.set_xscale("log")
        for _, row in case_df.iterrows():
            method_name = row["method"]
            ax.scatter(
                row["pipeline_time_mean"],
                100 * row["accuracy_mean"],
                s=95,
                color=METHOD_COLORS[method_name],
                alpha=0.9,
            )
            ax.annotate(
                METHOD_SHORT[method_name],
                (row["pipeline_time_mean"], 100 * row["accuracy_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=9,
            )
        ax.set_title(CASE_LABELS[case_name])
        ax.set_xlabel("Mean pipeline time (ms, log scale)")
        ax.grid(False)

    axes[0].set_ylabel("Mean accuracy (%)")
    fig.suptitle("Time vs Accuracy Tradeoff", fontsize=16)
    return fig
