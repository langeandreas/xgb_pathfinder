"""
Conditional Importance Score (CIS) and Scope Decider Score (SDS) for all features.
- CIS = mean|SHAP| x (1 + tanh(max(late_accel, 0))) x (1 + log1p(max(avg_fci, 0))), normalized to [0, 1]
- SDS = DIS x (1 + normalized_scope_entropy) x acceleration_boost x FCI_boost
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.colors import Normalize

from ..types import DecisionRule
from .tree_engine import TreeEngine


def mean_abs_shap(shap_values: Any) -> np.ndarray:
    """Return mean |SHAP| per feature from a (samples x features) SHAP output."""
    matrix = shap_values[-1] if isinstance(shap_values, list) else shap_values
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim == 3:
        matrix = matrix[:, :, -1]
    # abs -> mean avoids cancellation between positive and negative contributions.
    return np.mean(np.abs(matrix), axis=0)


def compute_gain_trajectories(
    booster: xgb.Booster,
    feature_names: list[str],
) -> dict[int, dict[str, Any]]:
    """Per-feature gain/cover series over the trees where the feature splits.

    Mirrors ``TreeAnalyzer.compute_trajectory_metrics`` (acceleration uses
    ``np.diff(gains, prepend=0)``), which differs from ``TrajectoryEngine``.
    """
    df_trees = booster.trees_to_dataframe()
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    name_to_idx.update(
        {f"f{i}": i for i in range(len(feature_names)) if f"f{i}" not in name_to_idx}
    )

    raw: dict[int, dict[str, list]] = defaultdict(
        lambda: {"iterations": [], "gains": [], "covers": [], "split_counts": []}
    )
    tree_count = int(df_trees["Tree"].max()) + 1 if not df_trees.empty else 0
    splits = df_trees[df_trees["Feature"] != "Leaf"].dropna(subset=["Feature"])
    # Combine repeated splits so each feature has one observation per active tree.
    # Trees without that feature are omitted, not represented by zero gain.
    grouped = splits.groupby(["Tree", "Feature"], sort=True).agg(
        gain=("Gain", "sum"), cover=("Cover", "sum"), n=("Gain", "size")
    )
    for (tree_idx, feature), row in grouped.iterrows(): # type: ignore
        idx = name_to_idx.get(str(feature))
        if idx is None:
            continue
        raw[idx]["iterations"].append(int(tree_idx))
        raw[idx]["gains"].append(float(row["gain"]))
        raw[idx]["covers"].append(float(row["cover"]))
        raw[idx]["split_counts"].append(int(row["n"]))

    out: dict[int, dict[str, Any]] = {}
    for idx, series in raw.items():
        active_iterations = np.asarray(series["iterations"], dtype=int)
        active_gains = np.asarray(series["gains"], dtype=float)
        active_covers = np.asarray(series["covers"], dtype=float)
        active_split_counts = np.asarray(series["split_counts"], dtype=float)

        # Missing feature uses have zero gain, so differences and the method's
        # early/late phases are measured on the full tree timeline.
        gains = np.zeros(tree_count, dtype=float)
        gains[active_iterations] = active_gains
        acceleration = np.diff(gains, prepend=0.0)
        # FCI is averaged only over trees where the feature is used (T_f).
        active_fci = active_covers / active_split_counts
        out[idx] = {
            "feature_name": feature_names[idx],
            "iterations": list(range(tree_count)),
            "active_iterations": series["iterations"],
            "velocity": gains.tolist(),
            "acceleration": acceleration.tolist(),
            "fci": active_fci.tolist(),
            "cumulative_gain": np.cumsum(gains).tolist(),
        }
    return out


def compute_decision_context(
    booster: xgb.Booster,
    feature_names: list[str],
    top_k: int = 20,
    rules: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per-feature decision-path context profiles (late-decider rate, scope features, impact)."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    tree_engine = TreeEngine(booster, feature_names)
    if rules is None:
        rules = tree_engine.extract_decision_rules_all_trees()
    tree_count = int(tree_engine.get_tree_count())

    per_feature: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "path_count": 0,
            "late_decider_count": 0,
            "early_presence_count": 0,
            "leaf_values_all_paths": [],
            "leaf_values_late_decider": [],
            "scope_counter": Counter(),
        }
    )
    all_leaf_values: list[float] = []

    # Each rule contributes one path observation, regardless of sample support.
    for rule in rules:
        path_features = [int(c["feature_index"]) for c in rule["conditions"]]
        if not path_features:
            continue
        leaf_value = float(rule["leaf_value"])
        all_leaf_values.append(leaf_value)
        path_len = len(path_features)

        # Count a feature once per path, using its first/last occurrence to
        # distinguish early scope-setting from late decision-making
        for feature_idx in sorted(set(path_features)):
            positions = [i for i, fi in enumerate(path_features) if fi == feature_idx]
            first_pos, last_pos = positions[0], positions[-1]
            stats = per_feature[feature_idx]
            stats["path_count"] += 1
            stats["leaf_values_all_paths"].append(leaf_value)
            if (first_pos + 1) / path_len <= 1.0 / 3.0:
                stats["early_presence_count"] += 1
            if (last_pos + 1) / path_len >= 2.0 / 3.0:
                stats["late_decider_count"] += 1
                stats["leaf_values_late_decider"].append(leaf_value)
                # Count distinct preceding features only on paths where this
                # feature acts as a late decider; a feature is not its own scope.
                for scope_idx in sorted(set(path_features[:last_pos]) - {feature_idx}):
                    stats["scope_counter"][scope_idx] += 1

    # Scale leaf magnitudes relative to the model's nonempty paths, avoid division by zero
    baseline = max(
        float(np.mean(np.abs(all_leaf_values))) if all_leaf_values else 0.0, 1e-9
    )

    profiles = []
    for feature_idx, stats in per_feature.items():
        path_count = int(stats["path_count"])
        leaf_all = np.asarray(stats["leaf_values_all_paths"], dtype=float)
        leaf_late = np.asarray(stats["leaf_values_late_decider"], dtype=float)
        late_decider_rate = stats["late_decider_count"] / path_count
        mean_leaf = float(np.mean(leaf_all)) if leaf_all.size else 0.0
        mean_leaf_late = float(np.mean(leaf_late)) if leaf_late.size else 0.0
        mean_abs_leaf_late = (
            float(np.mean(np.abs(leaf_late))) if leaf_late.size else 0.0
        )
        impact_ratio = mean_abs_leaf_late / baseline
        # Favor frequent late deciders with large leaves and a signed leaf mean
        # that differs from their broader path context
        decision_impact_score = (
            late_decider_rate * impact_ratio * (1.0 + abs(mean_leaf_late - mean_leaf))
        )

        scope_total = float(sum(stats["scope_counter"].values()))
        scope_entropy = 0.0
        # Low entropy means a few preceding features dominate the context.
        # Use all partners for entropy even though only top_k are retained below.
        if scope_total > 0:
            for cnt in stats["scope_counter"].values():
                p = cnt / scope_total
                scope_entropy -= p * float(np.log2(max(p, 1e-12)))
        scope_count = len(stats["scope_counter"])
        normalized_scope_entropy = (
            min(max(scope_entropy / float(np.log2(scope_count)), 0.0), 1.0)
            if scope_count > 1
            else 0.0
        )

        top_scope = sorted(
            stats["scope_counter"].items(), key=lambda kv: (-kv[1], kv[0])
        )[:top_k]
        profiles.append(
            {
                "feature_idx": int(feature_idx),
                "feature_name": feature_names[feature_idx],
                "path_count": path_count,
                "late_decider_rate": late_decider_rate,
                "early_presence_rate": stats["early_presence_count"] / path_count,
                "impact_magnitude_ratio": impact_ratio,
                "decision_impact_score": decision_impact_score,
                "scope_entropy": scope_entropy,
                "normalized_scope_entropy": normalized_scope_entropy,
                "top_scope_features": [
                    {
                        "feature_idx": int(si),
                        "cooccurrence_count": int(cnt),
                        "cooccurrence_rate": float(
                            cnt / stats["late_decider_count"]
                        ),
                    }
                    for si, cnt in top_scope
                ],
            },
        )

    profiles.sort(
        key=lambda p: (
            -p["decision_impact_score"],
            -p["late_decider_rate"],
            -p["path_count"],
            p["feature_idx"],
        )
    )
    return {"tree_count": tree_count, "feature_profiles": profiles}


def compute_cis(
    shap_by_feature: Mapping[int, float],
    trajectories: Mapping[int, Mapping[str, Any]],
) -> dict[int, float]:
    """Conditional Importance Score per feature index, normalized to [0, 1]."""
    out: dict[int, float] = {}
    for idx, metrics in trajectories.items():
        fci = np.asarray(metrics.get("fci", []), dtype=float)
        accel = np.asarray(metrics.get("acceleration", []), dtype=float)
        avg_fci = float(np.mean(fci)) if fci.size else 0.0
        late_accel = float(np.mean(accel[len(accel) // 2 :])) if accel.size else 0.0
        accel_boost = 1.0 + float(np.tanh(max(late_accel, 0.0)))
        fci_boost = 1.0 + float(np.log1p(max(avg_fci, 0.0)))
        out[int(idx)] = (
            float(shap_by_feature.get(int(idx), 0.0)) * accel_boost * fci_boost
        )
    max_val = max(out.values(), default=0.0)
    if max_val > 0.0:
        out = {key: value / max_val for key, value in out.items()}
    return out


def compute_sds(
    decision_context: Mapping[str, Any],
    trajectories: Mapping[int, Mapping[str, Any]],
) -> dict[int, float]:
    """Compute method-defined SDS using FCI as the uncertainty proxy."""
    out: dict[int, float] = {}
    for profile in decision_context.get("feature_profiles", []):
        idx = int(profile["feature_idx"])
        metrics = trajectories.get(idx, {})
        acceleration = np.asarray(metrics.get("acceleration", []), dtype=float)
        fci = np.asarray(metrics.get("fci", []), dtype=float)

        # Acceleration spans every tree; FCI is averaged over active trees only.
        late_accel = (
            float(np.mean(acceleration[len(acceleration) // 2 :]))
            if acceleration.size
            else 0.0
        )
        mean_fci = float(np.mean(fci)) if fci.size else 0.0
        acceleration_boost = 1.0 + float(np.tanh(max(late_accel, 0.0)))
        fci_boost = 1.0 + float(np.log1p(max(mean_fci, 0.0)))
        dis = float(profile.get("decision_impact_score", 0.0))
        normalized_entropy = float(profile.get("normalized_scope_entropy", 0.0))
        out[idx] = dis * (1.0 + normalized_entropy) * acceleration_boost * fci_boost
    return out


def compute_importance_scores(
    booster: xgb.Booster,
    feature_names: list[str],
    shap_values: Any,
    top_k_scope: int = 20,
    rules: list[DecisionRule] | None = None,
) -> pd.DataFrame:
    """Compute CIS and SDS for every feature.

    Returns one row per feature, sorted by CIS descending.
    """
    abs_shap = mean_abs_shap(shap_values)
    n_features = min(len(feature_names), abs_shap.size)
    shap_by_feature = {i: float(abs_shap[i]) for i in range(n_features)}

    trajectories = compute_gain_trajectories(booster, feature_names)
    decision_context = compute_decision_context(
        booster, feature_names, top_k=top_k_scope, rules=rules
    )
    cis = compute_cis(shap_by_feature, trajectories)
    sds = compute_sds(decision_context, trajectories)
    profiles = {int(p["feature_idx"]): p for p in decision_context["feature_profiles"]}

    rows = []
    # Keep every aligned feature, including unused ones; missing tree/path
    # evidence contributes zero rather than silently dropping the feature.
    for idx in range(n_features):
        profile = profiles.get(idx, {})
        rows.append(
            {
                "feature_idx": idx,
                "feature_name": feature_names[idx],
                "mean_abs_shap": shap_by_feature[idx],
                "conditional_importance_score": float(cis.get(idx, 0.0)),
                "scope_decider_score": float(sds.get(idx, 0.0)),
                "late_decider_rate": float(profile.get("late_decider_rate", 0.0)),
                "scope_entropy": float(profile.get("scope_entropy", 0.0)),
                "normalized_scope_entropy": float(
                    profile.get("normalized_scope_entropy", 0.0)
                ),
                "decision_impact_score": float(
                    profile.get("decision_impact_score", 0.0)
                ),
                "path_count": int(profile.get("path_count", 0)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("conditional_importance_score", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance_scores(
    scores_df: pd.DataFrame,
    top_n: int | None = None,
    feature_labels: Mapping[str, str] | None = None,
    sort_by: str = "cis",
    output_path: str | None = None,
):
    """Vertically stacked CIS (blue) and SDS (red) bar charts.

    :param scores_df: output of :func:`compute_importance_scores`
    :param top_n: keep only the top N features by ``sort_by``; all features if None
    :param feature_labels: optional feature_name -> readable label mapping
    :param sort_by: 'cis' or 'sds'
    :param output_path: save the figure when given
    :return: the matplotlib figure
    """
    sort_cols = {"cis": "conditional_importance_score", "sds": "scope_decider_score"}
    if sort_by not in sort_cols:
        raise ValueError(f"sort_by must be one of {sorted(sort_cols)}, got {sort_by!r}")

    plot_df = scores_df.sort_values(sort_cols[sort_by], ascending=False)
    if top_n is not None:
        plot_df = plot_df.head(int(top_n))
    # Both panels share the selected ordering so features can be compared directly.
    # Ascending so the highest-ranked feature appears at the top after barh.
    plot_df = plot_df.iloc[::-1]
    labels = [
        (feature_labels or {}).get(name, name)
        for name in plot_df["feature_name"].astype(str)
    ]

    n_features = len(plot_df)
    fig, (ax_cis, ax_sds) = plt.subplots(
        2, 1, figsize=(12, max(8.0, 0.76 * n_features + 2.0))
    )
    y_pos = np.arange(n_features)

    def _draw_bars(ax, values, cmap_name, xlabel, title, cbar_label):
        cmap = plt.get_cmap(cmap_name)
        max_v = max(values) if values else 1.0
        x_max = max_v * 1.22 + 0.02
        colors = [cmap(0.35 + 0.60 * (v / max(max_v, 1e-9))) for v in values]
        bars = ax.barh(y_pos, values, color=colors, edgecolor="none", height=0.7)
        for bar_iter, val in zip(bars, values):
            if val > 0.005:
                ax.text(
                    val + max_v * 0.01,
                    bar_iter.get_y() + bar_iter.get_height() / 2.0,
                    f"{val:.3f}",
                    va="center",
                    ha="left",
                    fontsize=12,
                    color="#333333",
                )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
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
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=max_v))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=12)

    _draw_bars(
        ax_cis,
        plot_df["conditional_importance_score"].tolist(),
        "Blues",
        "Conditional Importance Score (CIS, normalized)",
        "Conditional Importance Score (CIS)\nmean|SHAP| x accel-boost x FCI-boost",
        "CIS magnitude",
    )
    _draw_bars(
        ax_sds,
        plot_df["scope_decider_score"].tolist(),
        "Reds",
        "Scope Decider Score (SDS)",
        "Scope Decider Score (SDS)\ncontext-dep. x decision-likelihood x evidence",
        "SDS magnitude",
    )
    ax_cis.set_yticks(y_pos)
    ax_cis.set_yticklabels(labels, fontsize=12)
    ax_sds.set_yticks(y_pos)
    ax_sds.set_yticklabels(labels, fontsize=12)

    fig.suptitle(
        "Conditional Importance Score vs. Scope Decider Score\n"
        f"(features ordered by {sort_by.upper()})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def load_feature_labels(
    features_csv_path: str | Path,
    code_col: str = "codename",
    label_col: str = "explanation",
) -> dict[str, str]:
    """Build a feature_name -> readable label mapping from a lookup CSV."""
    lookup = pd.read_csv(features_csv_path).dropna(subset=[code_col, label_col])
    return dict(zip(lookup[code_col].astype(str), lookup[label_col].astype(str)))
