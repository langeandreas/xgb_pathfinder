"""Cohort metrics for households in a specific administrative region."""

from collections.abc import Hashable
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .visualizations import TITLE_FONT, _household_key, _normalise_name

_ORDER_COLUMNS = {
    "support": ["region_support", "region_mean_confidence"],
    "confidence": ["region_mean_confidence", "region_support"],
    "decision_impact": ["decision_impact", "region_support"],
}


def _select_region(hh_admin_df: pd.DataFrame, region: str | int) -> pd.DataFrame:
    """Rows of the admin mapping that belong to ``region`` (admin name or adm_code)."""
    if isinstance(region, (int, np.integer)) or str(region).strip().isdigit():
        selected = hh_admin_df[hh_admin_df["adm_code"].astype(int) == int(region)]
    else:
        selected = hh_admin_df[hh_admin_df["Name"].map(_normalise_name) == _normalise_name(region)]
    if selected.empty:
        raise ValueError(f"Region {region!r} not found in the household admin mapping.")
    return selected


def extract_region_cohorts(
    cohorts_df: pd.DataFrame,
    sample_to_cohorts: Mapping[Hashable, Sequence[str]],
    sample_ids: Sequence[Hashable],
    y: Sequence[float],
    predictions: Sequence[float],
    hh_admin_df: pd.DataFrame,
    region: str | int,
    min_support: int = 1,
    order_by: str = "support",
    top_n: int | None = None,
) -> pd.DataFrame:
    """Attach region-level metrics to cohorts calculated from regional samples.

    :param cohorts_df: cohorts calculated from the region's training samples
    :param sample_to_cohorts: sample id -> cohort ids the sample belongs to
    :param sample_ids: sample ids (household ids) aligned with ``y`` and ``predictions``
    :param y: observed targets aligned with ``sample_ids``
    :param predictions: model predictions aligned with ``sample_ids``
    :param hh_admin_df: output of ``load_household_admin_mapping``
    :param region: admin region name (case-insensitive) or ``adm_code``
    :param min_support: minimum region households required for a cohort
    :param order_by: 'support', 'confidence' or 'decision_impact'
    :param top_n: keep only the first N cohorts
    :return: cohort rows plus ``region_name``, ``adm_code``, ``region_households``,
        ``region_support``, ``region_share``, ``region_risk_rate``, ``region_mean_confidence``
    """
    if order_by not in _ORDER_COLUMNS:
        raise ValueError(f"order_by must be one of {sorted(_ORDER_COLUMNS)}, got {order_by!r}")

    admin = _select_region(hh_admin_df, region).copy()
    admin["household_id"] = _household_key(admin["household_id"]).values

    samples = pd.DataFrame(
        {
            "sample_id": list(sample_ids),
            "household_id": _household_key(sample_ids).values,
            "y": np.asarray(y, dtype=float).reshape(len(sample_ids), -1)[:, 0],
            "prediction": np.asarray(predictions, dtype=float).reshape(len(sample_ids), -1)[:, -1],
        }
    )
    region_samples = samples[samples["household_id"].isin(set(admin["household_id"]))]
    if region_samples.empty:
        raise ValueError(f"No samples fall in region {region!r}.")

    memberships = [
        {"household_id": row.household_id, "cohort_id": cohort_id, "y": row.y, "prediction": row.prediction}
        for row in region_samples.itertuples(index=False)
        for cohort_id in dict.fromkeys(sample_to_cohorts.get(row.sample_id, []))
    ]
    if not memberships:
        raise ValueError(f"No cohort samples fall in region {region!r}.")

    region_stats = (
        pd.DataFrame(memberships)
        .groupby("cohort_id", as_index=False)
        .agg(
            region_support=("household_id", "nunique"),
            region_risk_rate=("y", "mean"),
            region_mean_confidence=("prediction", "mean"),
        )
    )
    region_households = int(region_samples["household_id"].nunique())
    region_stats["region_share"] = region_stats["region_support"] / region_households

    out = cohorts_df.merge(region_stats, on="cohort_id", how="inner")
    out = out[out["region_support"] >= int(min_support)]
    out.insert(0, "region_name", str(admin["Name"].iloc[0]))
    out.insert(1, "adm_code", int(admin["adm_code"].iloc[0]))
    out.insert(2, "region_households", region_households)
    out = out.sort_values(_ORDER_COLUMNS[order_by], ascending=False).reset_index(drop=True)
    return out.head(int(top_n)) if top_n is not None else out


def plot_region_cohorts(
    region_df: pd.DataFrame,
    top_n: int = 10,
    title: str | None = None,
    output_path: str | None = None,
):
    """Horizontal bar chart of region household share per cohort.

    :param region_df: output of :func:`extract_region_cohorts`
    :param top_n: number of cohorts shown, in the order of ``region_df``
    :param title: overrides the generated title
    :param output_path: save the figure when given
    :return: the matplotlib figure
    """
    plot_df = region_df.head(int(top_n))
    region_name = str(plot_df["region_name"].iloc[0]) if not plot_df.empty else ""
    labels = [
        f"{row.cohort_id}\n{row.decider_feature_name} (risk {row.region_risk_rate:.0%})"
        for row in plot_df.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(12, 0.7 * max(len(plot_df), 3) + 1.5))
    ax.barh(labels, plot_df["region_share"], color="orange", alpha=0.7, edgecolor="grey")
    for y_pos, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            row.region_share, # type: ignore
            y_pos,
            f" {row.region_support} hh",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )
    ax.invert_yaxis()
    ax.set_xlabel("Share of region households", fontsize=12, fontweight="bold")
    ax.set_title(title or f"Cohorts in {region_name}", loc="left", pad=10, fontdict=TITLE_FONT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
