"""Geographic visualisation of tree-derived cohort and decision-path insights.

Projects cohorts produced by the attribution analysis pipeline
onto administrative polygons.

Pipeline::

    load_admin_geodata          -> polygons (Name, geometry)
    load_household_admin_mapping-> household_id -> admin region
    build_cohort_membership     -> household_id -> cohort_id (full dataset)
    aggregate_cohorts_by_region -> per-region metric
    visualise_cohort_map        -> choropleth
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from textwrap import fill
from typing import Any, TypeVar

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from shapely import wkt

from ..analysis.cohort_engine import build_cohort_membership
from ..types import RegionImpactMetric
from .geo_plotter import GeoPlotter
from .impact_aggregator import aggregate_cohorts_by_region

SampleId = TypeVar("SampleId", bound=Hashable)
MembershipSampleId = TypeVar("MembershipSampleId", bound=Hashable)

__all__ = [
    "aggregate_cohorts_by_region",
    "build_cohort_membership",
    "load_admin_geodata",
    "load_household_admin_mapping",
    "plot_cohort_geography",
    "visualise_cohort_map",
    "visualise_region_decision_paths",
]

TITLE_FONT = {"fontsize": "17"}

_ADMIN_COLUMNS = {
    "adm1": {"geometry": "adm1Geometry", "code": "adm1Code", "name": "adm1Name"},
    "adm2": {"geometry": "adm2Geometry", "code": "adm2Code", "name": "adm2Name_x"},
}


def _normalise_name(value: Any) -> str:
    return str(value).strip().casefold()


def _household_key(values: Any) -> pd.Series:
    """Household ids are string-like in some countries and integer in others."""
    return pd.Series(values).astype(str).str.strip()


# geo loaders


def load_admin_geodata(
    geodata_path: str, admin_level: str = "adm2"
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of admin polygons from a WKT geodata CSV.

    :param geodata_path: path to a geodata CSV holding WKT polygons per admin unit
    :param admin_level: 'adm1' or 'adm2'
    :return: GeoDataFrame with columns ``adm_code``, ``Name``, ``geometry`` (EPSG:4326)
    """
    if admin_level not in _ADMIN_COLUMNS:
        raise ValueError(
            f"admin_level must be one of {sorted(_ADMIN_COLUMNS)}, got {admin_level!r}"
        )

    cols = _ADMIN_COLUMNS[admin_level]
    raw = pd.read_csv(geodata_path)

    name_col = cols["name"]
    if name_col not in raw.columns:  # adm2Name_x only exists in the merged export
        name_col = cols["name"].replace("_x", "")
    missing = [
        c for c in (cols["geometry"], cols["code"], name_col) if c not in raw.columns
    ]
    if missing:
        raise KeyError(f"{geodata_path} is missing required columns: {missing}")

    geo = raw[[cols["code"], name_col, cols["geometry"]]].copy()
    geo.columns = ["adm_code", "Name", "wkt"]
    geo = geo.dropna(subset=["wkt", "adm_code"]).drop_duplicates(subset=["adm_code"])
    geo["geometry"] = geo["wkt"].map(wkt.loads)
    geo = geo.drop(columns=["wkt"])
    geo["adm_code"] = geo["adm_code"].astype(int)
    geo["Name"] = geo["Name"].astype(str).str.strip()

    return gpd.GeoDataFrame(geo, geometry="geometry", crs="EPSG:4326").reset_index(
        drop=True
    )


def load_household_admin_mapping(
    mapping_path: str,
    admin_level: str = "adm2",
    household_col: str = "household_id",
) -> pd.DataFrame:
    """Load the household -> admin region lookup.

    :param mapping_path: path to the household/geography mapping CSV
    :param admin_level: 'adm1' or 'adm2'
    :param household_col: household identifier column in the mapping CSV
    :return: DataFrame with columns ``household_id``, ``adm_code``, ``Name``
    """
    if admin_level not in _ADMIN_COLUMNS:
        raise ValueError(
            f"admin_level must be one of {sorted(_ADMIN_COLUMNS)}, got {admin_level!r}"
        )

    code_col = _ADMIN_COLUMNS[admin_level]["code"]
    name_col = _ADMIN_COLUMNS[admin_level]["name"].replace("_x", "")
    raw = pd.read_csv(mapping_path)

    missing = [c for c in (household_col, code_col, name_col) if c not in raw.columns]
    if missing:
        raise KeyError(f"{mapping_path} is missing required columns: {missing}")

    mapping = raw[[household_col, code_col, name_col]].copy()
    mapping.columns = ["household_id", "adm_code", "Name"]
    mapping = mapping.dropna().drop_duplicates(subset=["household_id"])
    mapping["household_id"] = _household_key(mapping["household_id"])
    mapping["adm_code"] = mapping["adm_code"].astype(int)
    mapping["Name"] = mapping["Name"].astype(str).str.strip()

    return mapping.reset_index(drop=True)


def visualise_cohort_map(
    region_aggregation: dict[str, RegionImpactMetric],
    geodata: gpd.GeoDataFrame | str,
    output_path: str | None = None,
    color_by: str = "total_impact",
    admin_level: int = 2,
    **plot_kwargs: Any,
) -> Figure:
    """Plot existing regional impact metrics using :class:`GeoPlotter`.

    ``geodata`` is a GeoDataFrame or a path to GeoJSON, Shapefile, or the WKT
    administrative CSV. Additional options go to the static choropleth plotter.
    """
    if not region_aggregation:
        raise ValueError("No regional cohort evidence to plot")
    plotter = GeoPlotter(region_aggregation)
    if isinstance(geodata, str):
        plotter.load_geodata(geodata, admin_level=admin_level)
    else:
        plotter.geodata = geodata
    return plotter.plot_choropleth_matplotlib(
        output_path=output_path, color_by=color_by, **plot_kwargs
    )


def plot_cohort_geography(
    cohorts: pd.DataFrame,
    sample_region_mapping: pd.DataFrame | Mapping[SampleId, str],
    sample_to_cohort_mapping: Mapping[MembershipSampleId, Sequence[str]],
    geodata: gpd.GeoDataFrame | str,
    impact_metric: str = "mean_member_impact",
    output_path: str | None = None,
    admin_level: int = 2,
    **plot_kwargs: Any,
) -> Figure:
    """Aggregate evaluated memberships and plot the selected regional metric.

    Uses the same metrics as ``ModelExplainer.aggregate_by_region``.
    ``sample_region_mapping`` also accepts ``load_household_admin_mapping`` output.
    """
    aggregation = aggregate_cohorts_by_region(
        cohorts,
        sample_region_mapping,
        sample_to_cohort_mapping,
        impact_metric=impact_metric,
        config={"admin_level": admin_level},
    )
    return visualise_cohort_map(
        aggregation,
        geodata,
        output_path=output_path,
        admin_level=admin_level,
        **plot_kwargs,
    )


def visualise_region_decision_paths(
    region_df: pd.DataFrame,
    top_n: int = 10,
    title: str | None = None,
    output_path: str | None = None,
) -> Figure:
    """Show regional cohort shares alongside their scopes and decider evidence.

    Accepts ``extract_region_cohorts`` output, preserving its ranking. When
    available, ``decider_prediction`` supplies the actual split conditions;
    otherwise the decider feature is shown without inventing a threshold.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if region_df.empty:
        raise ValueError("No regional cohorts to plot")
    plot_df = region_df.head(top_n)
    labels = []
    for row in plot_df.to_dict(orient="records"):
        scope = str(row["scope_signature"])
        decider = row.get("decider_prediction") or row["decider_feature_name"]
        labels.append(
            f"{row['cohort_id']}\n"
            + fill(f"Scope: {scope}", width=65)
            + "\n"
            + fill(f"Decider: {decider}", width=65)
        )

    heights = [max(1.0, 0.24 * (label.count("\n") + 1)) for label in labels]
    positions = np.cumsum(heights) - np.asarray(heights) / 2
    fig, ax = plt.subplots(figsize=(14, float(sum(heights)) + 2))
    ax.barh(positions, plot_df["region_share"], height=0.6, color="orange", alpha=0.7)
    ax.set_yticks(positions, labels=labels, fontsize=9)
    for position, row in zip(positions, plot_df.to_dict(orient="records")):
        ax.text(
            row["region_share"],
            position,
            f" {row['region_support']} hh; "
            f"mean prediction {row['region_mean_confidence']:.0%}",
            va="center",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.4)
    ax.set_xlabel("Share of region households (cohorts may overlap)")
    ax.set_title(
        title or f"Decision paths in {plot_df['region_name'].iloc[0]}",
        loc="left",
        fontdict=TITLE_FONT,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
