"""Plotting helpers for xgb-pathfinder feature analysis."""

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def plot_importance_scores(
    scores_df: pd.DataFrame,
    top_n: int | None = None,
    feature_labels: Mapping[str, str] | None = None,
    sort_by: str = "cis",
    output_path: str | Path | None = None,
):
    """Plot all features' CIS and SDS as vertically stacked bar charts.

    The styling mirrors the analysis app's Feature Story plot, with CIS (blue)
    above SDS (red), a shared feature order, horizontal bars, value labels, row
    separators, and color bars. By default all features are shown, ordered by
    CIS descending (highest at the top).

    Parameters
    ----------
    scores_df : pandas.DataFrame
        Output of ``compute_importance_scores``. Must contain ``feature_name``,
        ``conditional_importance_score``, and ``scope_decider_score`` columns.
    top_n : int, optional
        If provided, show only the top N features according to ``sort_by``.
    feature_labels : mapping, optional
        Mapping from feature names to display labels.
    sort_by : {"cis", "sds"}, default="cis"
        Metric used to rank features. The same order is used in both panels.
    output_path : str or pathlib.Path, optional
        Save the figure to this path when provided.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure. The caller is responsible for closing it when
        running outside an interactive plotting environment.
    """
    sort_columns = {
        "cis": "conditional_importance_score",
        "sds": "scope_decider_score",
    }
    if sort_by not in sort_columns:
        raise ValueError(
            f"sort_by must be one of {sorted(sort_columns)}, got {sort_by!r}"
        )

    required = {"feature_name", *sort_columns.values()}
    missing = required.difference(scores_df.columns)
    if missing:
        raise ValueError(f"scores_df is missing required columns: {sorted(missing)}")
    if top_n is not None and int(top_n) < 1:
        raise ValueError("top_n must be a positive integer or None")

    plot_df = scores_df.copy()
    for column in sort_columns.values():
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0)
    plot_df = plot_df.sort_values(sort_columns[sort_by], ascending=False, kind="stable")
    if top_n is not None:
        plot_df = plot_df.head(int(top_n))
    # barh displays the final row at the top
    #  reverse to put the highest first.
    plot_df = plot_df.iloc[::-1]

    labels = [
        (feature_labels or {}).get(name, name)
        for name in plot_df["feature_name"].astype(str)
    ]
    cis_values = plot_df["conditional_importance_score"].astype(float).tolist()
    sds_values = plot_df["scope_decider_score"].astype(float).tolist()

    n_features = len(plot_df)
    fig, (ax_cis, ax_sds) = plt.subplots(
        2, 1, figsize=(12, max(8.0, 0.76 * n_features + 2.0))
    )
    y_pos = np.arange(n_features)

    def draw_bars(ax, values, cmap_name, xlabel, title, colorbar_label):
        cmap = plt.get_cmap(cmap_name)
        max_value = max(values, default=0.0)
        color_scale = max(max_value, 1e-9)
        x_max = max_value * 1.22 + 0.02
        colors = [cmap(0.35 + 0.60 * (value / color_scale)) for value in values]
        bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.7)

        for bar_iter, value in zip(bars, values):
            if value > 0.005:
                ax.text(
                    value + max_value * 0.01,
                    bar_iter.get_y() + bar_iter.get_height() / 2.0,
                    f"{value:.3f}",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#333333",
                )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(0.0, x_max)
        ax.axvline(x=0.0, color="#aaaaaa", linewidth=0.8)
        ax.hlines(
            np.arange(-0.5, len(values) + 0.5, 1.0),
            xmin=0.0,
            xmax=x_max,
            colors="#d6d6d6",
            linewidth=0.6,
            alpha=0.8,
            zorder=0,
        )
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=cmap,
                norm=Normalize(vmin=0.0, vmax=color_scale),
            ),
            ax=ax,
            fraction=0.025,
            pad=0.03,
        )
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    draw_bars(
        ax_cis,
        cis_values,
        "Blues",
        "Conditional Importance Score (CIS, normalized)",
        "Conditional Importance Score (CIS)\nmean|SHAP| x accel-boost x FCI-boost",
        "CIS magnitude",
    )
    draw_bars(
        ax_sds,
        sds_values,
        "Reds",
        "Scope Decider Score (SDS)",
        "Scope Decider Score (SDS)\ncontext-dep. x decision-likelihood x evidence",
        "SDS magnitude",
    )
    ax_cis.set_yticks(y_pos)
    ax_cis.set_yticklabels(labels, fontsize=11)
    ax_sds.set_yticks(y_pos)
    ax_sds.set_yticklabels(labels, fontsize=11)
    fig.suptitle(
        "Conditional Importance Score vs. Scope Decider Score\n"
        f"(features ordered by {sort_by.upper()})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


__all__ = ["plot_importance_scores"]
