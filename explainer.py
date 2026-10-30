"""
xgb-pathfinder: Generalized tree-based model explanation package.

Main entry point: ModelExplainer class for XGBoost tree analysis.
Computes trajectory metrics, extracts cohorts, and provides geographic visualization.
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from .analysis.cohort_engine import CohortEngine, build_cohort_membership
from .analysis.feature_analyzer import FeatureAnalyzer
from .analysis.importance_scores import (
    compute_importance_scores as _compute_importance_scores,
)
from .analysis.importance_scores import (
    plot_importance_scores as _plot_importance_scores,
)
from .analysis.shap_engine import ShapEngine
from .analysis.trajectory_engine import TrajectoryEngine
from .config import DEFAULT_CONFIG
from .geographic.geo_plotter import GeoPlotter
from .geographic.impact_aggregator import aggregate_cohorts_by_region
from .geographic.region_cohorts import (
    _select_region,
)
from .geographic.region_cohorts import (
    extract_region_cohorts as _extract_region_cohorts,
)
from .geographic.region_cohorts import (
    plot_region_cohorts as _plot_region_cohorts,
)
from .geographic.visualizations import _household_key, load_household_admin_mapping
from .types import (
    CohortRecord,
    DecisionRule,
    FeatureProfile,
    FeatureTrajectory,
    PathfinderConfig,
    RegionImpactMetric,
)


class ModelExplainer:
    """
    Main entry point for XGBoost model explanation using xgb-pathfinder.

    Computes trajectory metrics (velocity, acceleration, FCI) across training iterations,
    extracts decision-based cohorts, and provides geographic visualization of impact metrics.

    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster model with training history.
    x_train : array-like, shape (n_samples, n_features)
        Training feature matrix used for SHAP computation.
    y_train : array-like, shape (n_samples,)
        Training target values.
    feature_names : list[str], optional
        Human-readable feature names. If None, uses ["feature_0", "feature_1", ...].
    config : PathfinderConfig, optional
        Configuration dict for thresholds, top-K values, etc.
        Uses DEFAULT_CONFIG if not provided.

    Attributes
    ----------
    booster : xgboost.Booster
        The fitted model.
    x_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training targets.
    feature_names : list[str]
        Feature name mapping.
    config : PathfinderConfig
        Active configuration.
    Notes
    -----
    Analysis results are computed lazily and cached until the cache is cleared.
    """

    def __init__(
        self,
        booster: xgb.Booster,
        x_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
        config: PathfinderConfig | None = None,
        sample_ids: Any | None = None,
    ):
        """Initialize explainer with model and data."""
        self.booster = booster
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        self.sample_ids = np.asarray(
            x_train.index
            if sample_ids is None and isinstance(x_train, pd.DataFrame)
            else (np.arange(len(self.x_train)) if sample_ids is None else sample_ids)
        )
        if len(self.sample_ids) != len(self.x_train):
            raise ValueError(
                "sample_ids length must match the number of training samples"
            )

        # Set feature names
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(self.x_train.shape[1])]
        else:
            if len(feature_names) != self.x_train.shape[1]:
                raise ValueError(
                    f"feature_names length ({len(feature_names)}) does not match "
                    f"number of features ({self.x_train.shape[1]})"
                )
            self.feature_names = list(feature_names)

        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self._importance_scores: dict[int, pd.DataFrame] = {}
        self._geographic_aggregation: dict[str, RegionImpactMetric] | None = None
        self._trajectory_engine = TrajectoryEngine(
            self.booster, self.feature_names, self.config
        )
        self._cohort_engine = CohortEngine(
            self.booster, self.x_train, self.y_train, self.feature_names, self.config
        )
        self._feature_analyzer = FeatureAnalyzer(
            self.booster, self.x_train, self.y_train, self.feature_names, self.config
        )
        self._shap_engine = ShapEngine(
            self.booster, self.x_train, self.feature_names, self.config
        )

    # ========================================================================
    # Public API: Core Analysis
    # ========================================================================

    def compute_trajectory_metrics(self) -> dict[str, FeatureTrajectory]:
        """
        Compute per-feature trajectory metrics across training iterations.

        Returns
        -------
        dict[str, FeatureTrajectory]
            Maps feature name -> trajectory (velocity, acceleration, FCI, etc. per iteration).

        Examples
        --------
        >>> trajectories = explainer.compute_trajectory_metrics()
        >>> for feature_name, traj in trajectories.items():
        ...     print(f"{feature_name}: peak velocity = {traj['peak_velocity']}")
        """
        return self._trajectory_metrics

    def extract_cohorts(
        self,
        min_support: int | None = None,
        include_feature_profiles: bool = False,
        include_shap: bool = False,
    ) -> pd.DataFrame:
        """
        Extract cohorts (population segments) from decision tree paths.

        A cohort is defined by:
        - scope_signature: AND-ed conditions on early scope features
        - decider_feature: Primary decision feature
        - Enriched with metrics: support, risk rate, decision impact, SHAP summary

        Parameters
        ----------
        min_support : int, optional
            Minimum sample count per cohort. Uses config value if not specified.
        include_feature_profiles : bool
            If True, compute feature profiles for samples in each cohort (slower).
        include_shap : bool
            If True, compute local SHAP feature summaries for each cohort (slower).

        Returns
        -------
        pd.DataFrame
            One row per cohort, sorted by decision_impact (highest first).

        Examples
        --------
        >>> cohorts = explainer.extract_cohorts(min_support=20)
        >>> cohorts.head()
        """
        cohort_records = self._cohort_engine.build_cohorts(
            min_support=min_support,
            include_shap=include_shap,
        )
        cohorts = pd.DataFrame.from_records(
            cohort_records,
            columns=list(CohortRecord.__annotations__),
        )
        self.__dict__.update(
            _cohorts=cohorts,
            _decision_rules=self._cohort_engine.extract_decision_rules(),
        )
        if include_feature_profiles:
            self.get_feature_profiles()
        return cohorts

    def get_feature_profiles(self) -> dict[str, FeatureProfile]:
        """
        Get comprehensive per-feature profiles.

        Combines trajectory metrics, SHAP decision impact, feature interactions,
        and path statistics into a single profile per feature.

        Returns
        -------
        dict[str, FeatureProfile]
            Maps feature name -> profile.

        Examples
        --------
        >>> profiles = explainer.get_feature_profiles()
        >>> for feature_name, profile in profiles.items():
        ...     print(f"{feature_name}: behavior={profile['behavior_signature']}, "
        ...           f"peak_impact={profile['decision_impact']['total_swing']:.3f}")
        """
        return self._feature_profiles

    def get_segments(
        self,
        rules: list[DecisionRule] | None = None,
    ) -> pd.DataFrame:
        """
        Assign samples to cohorts/segments based on decision rules.

        Parameters
        ----------
        rules : list[DecisionRule], optional
            Decision rules to evaluate. If None, uses rules from extract_cohorts().

        Returns
        -------
        pd.DataFrame
            One row per sample, columns: sample_index, assigned_cohort_ids (list),
            primary_cohort_id, confidence.

        Examples
        --------
        >>> segments = explainer.get_segments()
        >>> cohort_counts = segments['primary_cohort_id'].value_counts()
        """
        assignments = self._cohort_engine.assign_samples_to_rules().copy()
        default_rules = self._decision_rules
        rule_ids = {rule["rule_id"] for rule in (rules or default_rules)}
        assignments["assigned_rule_ids"] = assignments["assigned_rule_ids"].apply(
            lambda ids: [rule_id for rule_id in ids if rule_id in rule_ids]
        )
        assignments["primary_rule_id"] = assignments["assigned_rule_ids"].apply(
            lambda ids: ids[0] if ids else None
        )
        assignments["confidence"] = self._predict()
        return assignments

    def compute_importance_scores(self, top_k_scope: int = 20) -> pd.DataFrame:
        """
        Compute Conditional Importance Score (CIS) and Scope Decider Score (SDS) for all features.

        Same formulas as the analysis app's Feature Story page.

        Parameters
        ----------
        top_k_scope : int
            Scope features retained per feature for the SDS evidence terms.

        Returns
        -------
        pd.DataFrame
            One row per feature, sorted by CIS descending.
        """
        if top_k_scope not in self._importance_scores:
            self._importance_scores[top_k_scope] = _compute_importance_scores(
                self.booster,
                self.feature_names,
                self._shap_engine.compute_shap_values(),
                top_k_scope=top_k_scope,
                rules=self._decision_rules,
            )
        return self._importance_scores[top_k_scope].copy()

    def plot_importance_scores(self, top_k_scope: int = 20, **plot_kwargs) -> Any:
        """
        Plot CIS and SDS side by side.

        ``plot_kwargs`` are passed to ``plot_importance_scores``: ``top_n``,
        ``feature_labels``, ``sort_by`` ('cis' or 'sds'), ``output_path``.
        """
        return _plot_importance_scores(
            self.compute_importance_scores(top_k_scope=top_k_scope), **plot_kwargs
        )

    # ========================================================================
    # Public API: Geographic & Impact
    # ========================================================================

    def aggregate_by_region(
        self,
        sample_region_mapping: pd.DataFrame | dict[int, str],
        admin_level: int | None = None,
        impact_metric: str = "mean_member_impact",
    ) -> dict[str, RegionImpactMetric]:
        """
        Summarize the magnitude of cohort decision paths represented in each region.

        Cohorts are post-hoc groups of tree paths, not model inputs. These metrics
        describe path magnitude and representation; they do not measure causal
        dependence of predictions on a cohort.

        Parameters
        ----------
        geodata_path : str
            Path to GeoJSON or Shapefile with polygon geometries.
        sample_region_mapping : pd.DataFrame or dict
            Maps sample index or ID to region name.
            If DataFrame: index must be sample index, values must be region names.
            If dict: keys are sample index, values are region names.
        admin_level : int, optional
            Administrative hierarchy level (1, 2, etc.). Uses config if not specified.
        impact_metric : str
            ``mean_member_impact`` (default) averages absolute leaf magnitude over
            sample-cohort memberships. ``cumulative_member_impact`` sums those
            magnitudes and is population-sensitive. ``distinct_cohort_count`` counts
            represented cohorts.

        Returns
        -------
        dict[str, RegionImpactMetric]
            Maps region name -> impact metrics (sorted by total_impact descending).
        """
        cohorts = self.extract_cohorts()
        sample_to_cohorts = self._sample_to_cohorts(cohorts)
        aggregation_config = dict(self.config)
        if admin_level is not None:
            aggregation_config["admin_level"] = admin_level
        aggregation = aggregate_cohorts_by_region(
            cohorts,
            sample_region_mapping,
            sample_to_cohort_mapping=sample_to_cohorts,
            impact_metric=impact_metric,
            config=aggregation_config,
        )
        self._geographic_aggregation = aggregation
        return aggregation

    def plot_geographic_impact(
        self,
        geodata_path: str,
        sample_region_mapping: pd.DataFrame | dict[int, str],
        output_path: str | None = None,
        impact_metric: str = "mean_member_impact",
        **plot_kwargs,
    ) -> Any:
        """
        Map a clearly labeled regional cohort path metric.

        Parameters
        ----------
        geodata_path : str
            Path to GeoJSON or Shapefile.
        sample_region_mapping : pd.DataFrame or dict
            Sample to region mapping.
        output_path : str, optional
            Save map to file. If None, returns matplotlib/folium object.
        impact_metric : str
            Regional metric selected as ``total_impact``: ``mean_member_impact``,
            ``cumulative_member_impact``, or ``distinct_cohort_count``.
        **plot_kwargs
            ``color_by`` may be ``total_impact``, ``mean_member_impact``,
            ``cumulative_member_impact``, ``cohort_count``, ``sample_count``,
            ``membership_count``, ``avg_impact_per_cohort``, or
            ``avg_impact_per_sample``. Static maps also accept ``title``,
            ``figsize``, ``cmap``, ``edgecolor``, and ``linewidth``. Interactive
            maps accept ``title``, ``zoom_start``, ``fill_color``, and ``tiles``.

        Returns
        -------
        Any
            Matplotlib Figure.

        Examples
        --------
        >>> fig = explainer.plot_geographic_impact(
        ...     geodata_path="admin_boundaries.geojson",
        ...     sample_region_mapping=mapping,
        ...     impact_metric="mean_member_impact",
        ...     cmap="viridis",
        ... )
        """
        admin_level = plot_kwargs.pop("admin_level", None)
        aggregation = self.aggregate_by_region(
            sample_region_mapping, admin_level=admin_level, impact_metric=impact_metric
        )
        plotter = GeoPlotter(aggregation)
        plotter.load_geodata(geodata_path, admin_level=admin_level)
        color_by = plot_kwargs.pop("color_by", "total_impact")
        
        return plotter.plot_choropleth_matplotlib(
            output_path=output_path, color_by=color_by, **plot_kwargs
        )

    def extract_region_cohorts(
        self,
        region: str | int,
        mapping_path: str | None = None,
        hh_admin_df: pd.DataFrame | None = None,
        admin_level: str = "adm2",
        min_cohort_support: int | None = None,
        min_region_support: int = 1,
        order_by: str = "support",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Recalculate cohorts using only households in one administrative region.

        Samples are matched to households through ``sample_ids``. The model's
        decision rules are retained, but cohort support and metrics are computed
        only from the selected region's training samples.

        Parameters
        ----------
        region : str or int
            Admin region name (case-insensitive) or ``adm_code``.
        mapping_path : str, optional
            Household -> admin mapping CSV (e.g. ``file_name.csv``).
        hh_admin_df : pd.DataFrame, optional
            Pre-loaded output of ``load_household_admin_mapping``; overrides ``mapping_path``.
        admin_level : str
            'adm1' or 'adm2' (used with ``mapping_path``).
        min_cohort_support : int, optional
            Minimum regional sample support for cohort extraction. Uses config if not specified.
        min_region_support : int
            Minimum region households required for a cohort.
        order_by : str
            'support', 'confidence' or 'decision_impact'.
        top_n : int, optional
            Keep only the first N cohorts.

        Returns
        -------
        pd.DataFrame
            Cohort rows with region-level support, share, risk rate and mean confidence.
        """
        if hh_admin_df is None:
            if mapping_path is None:
                raise ValueError("Provide either mapping_path or hh_admin_df.")
            hh_admin_df = load_household_admin_mapping(
                mapping_path, admin_level=admin_level
            )

        admin = _select_region(hh_admin_df, region)
        region_households = set(_household_key(admin["household_id"]))
        region_mask = _household_key(self.sample_ids).isin(region_households).to_numpy()
        region_indices = np.flatnonzero(region_mask)
        if len(region_indices) == 0:
            raise ValueError(f"No samples fall in region {region!r}.")

        region_x = self.x_train[region_indices]
        region_y = self.y_train[region_indices]
        region_engine = CohortEngine(
            self.booster, region_x, region_y, self.feature_names, self.config
        )
        cohort_records = region_engine.build_cohorts(min_support=min_cohort_support)
        cohorts = pd.DataFrame.from_records(
            cohort_records,
            columns=list(CohortRecord.__annotations__),
        )
        assignments = region_engine.assign_samples_to_rules()
        rule_sample_indices = {}
        for row in assignments.itertuples(index=False):
            for rule_id in row.assigned_rule_ids:  # type: ignore
                rule_sample_indices.setdefault(rule_id, set()).add(
                    int(row.sample_index)  # type: ignore
                )

        sample_to_cohorts = build_cohort_membership(
            cohorts, assignments, self.sample_ids[region_indices]
        )
        region_predictions = self.booster.predict(
            xgb.DMatrix(region_x, feature_names=self.booster.feature_names)
        )
        rules_by_id = {
            rule["rule_id"]: rule for rule in region_engine.extract_decision_rules()
        }
        positive_predictions = region_predictions >= 0.5

        def format_decider_prediction(cohort) -> str:
            criterion_samples = {}
            for rule_id in cohort.contributing_rules:
                conditions = rules_by_id[rule_id]["conditions"]
                criterion = (
                    f"{conditions[-1]['feature_name']} "
                    f"{conditions[-1]['operator']} "
                    f"{conditions[-1]['threshold']:g}"
                    if conditions
                    else "root"
                )
                criterion_samples.setdefault(criterion, set()).update(
                    rule_sample_indices.get(rule_id, set())
                )

            predictions = []
            for criterion, sample_indices in criterion_samples.items():
                indices = sorted(sample_indices)
                positive_share = float(np.mean(positive_predictions[indices]))
                predictions.append(
                    f"{criterion} -> positive prediction in {positive_share:.0%} "
                    f"(n={len(indices)})"
                )
            return "; ".join(predictions)

        if not cohorts.empty:
            cohorts["decider_prediction"] = cohorts.apply(
                format_decider_prediction, axis=1
            )

        return _extract_region_cohorts(
            cohorts,
            sample_to_cohorts,
            self.sample_ids[region_indices],
            region_y,
            region_predictions,
            hh_admin_df,
            region,
            min_support=min_region_support,
            order_by=order_by,
            top_n=top_n,
        )

    def plot_region_cohorts(
        self,
        region: str | int,
        top_n: int = 10,
        title: str | None = None,
        output_path: str | None = None,
        **extract_kwargs,
    ) -> Any:
        """Bar chart of the cohorts in one region. 
        ``extract_kwargs`` go to ``extract_region_cohorts``."""
        region_df = self.extract_region_cohorts(region, **extract_kwargs)
        return _plot_region_cohorts(
            region_df, top_n=top_n, title=title, output_path=output_path
        )

    def _predict(self) -> np.ndarray:
        return self.booster.predict(
            xgb.DMatrix(self.x_train, feature_names=self.booster.feature_names)
        )

    def _sample_to_cohorts(self, cohorts: pd.DataFrame) -> dict[Any, list[str]]:
        """Map each sample id to the cohort ids whose rules it satisfies."""
        return build_cohort_membership(cohorts, self.get_segments(), self.sample_ids)

    # ========================================================================
    # Public API: Export & Serialization
    # ========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Export all computed metrics as a nested dictionary.

        Suitable for serialization to JSON or further processing.

        Returns
        -------
        dict[str, Any]
            Keys: "trajectory_metrics", "feature_profiles", "cohorts", "decision_rules".

        Examples
        --------
        >>> data = explainer.to_dict()
        >>> import json
        >>> with open("results.json", "w") as f:
        ...     json.dump(data, f, indent=2)
        """
        payload = {
            "config": self.config,
            "feature_names": self.feature_names,
            "trajectory_metrics": self._trajectory_metrics,
            "feature_profiles": self._feature_profiles,
            "decision_rules": self._decision_rules,
            "cohorts": self._cohorts.to_dict(orient="records"),
        }
        if self._geographic_aggregation is not None:
            payload["geographic_aggregation"] = self._geographic_aggregation
        return self._json_safe(payload)

    def to_json(self, output_path: str, indent: int | None = None) -> None:
        """
        Export all metrics to JSON file.

        Parameters
        ----------
        output_path : str
            Path to output JSON file.
        indent : int, optional
            JSON indentation. Uses config if not specified.
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as output_file:
            json.dump(
                self.to_dict(),
                output_file,
                indent=int(self.config["json_indent"]) if indent is None else indent, #type: ignore
            )

    def to_csv(self, output_dir: str) -> None:
        """
        Export metrics to separate CSV files in a directory.

        Creates: trajectory_metrics.csv, cohorts.csv, feature_profiles.csv, etc.

        Parameters
        ----------
        output_dir : str
            Directory to write CSV files.
        """
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        trajectory_rows = []
        for feature_name, trajectory in payload["trajectory_metrics"].items():
            for metric in trajectory["metrics"]:
                trajectory_rows.append({"feature_name": feature_name, **metric})
        pd.DataFrame(trajectory_rows).to_csv(
            target / "trajectory_metrics.csv", index=False
        )
        pd.DataFrame(payload["cohorts"]).to_csv(target / "cohorts.csv", index=False)
        pd.DataFrame(payload["decision_rules"]).to_csv(
            target / "decision_rules.csv", index=False
        )
        profile_rows = []
        for feature_name, profile in payload["feature_profiles"].items():
            row = {
                "feature_name": feature_name,
                "behavior_signature": profile["behavior_signature"],
            }
            row.update(profile["decision_impact"])
            row.update(profile["path_statistics"])
            profile_rows.append(row)
        pd.DataFrame(profile_rows).to_csv(target / "feature_profiles.csv", index=False)

    # ========================================================================
    # Private: State Management
    # ========================================================================

    @cached_property
    def _trajectory_metrics(self) -> dict[str, FeatureTrajectory]:
        return self._trajectory_engine.compute_trajectories()

    @cached_property
    def _feature_profiles(self) -> dict[str, FeatureProfile]:
        return self._feature_analyzer.build_profiles()

    @cached_property
    def _cohorts(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            self._cohort_engine.build_cohorts(),
            columns=list(CohortRecord.__annotations__),
        )

    @cached_property
    def _decision_rules(self) -> list[DecisionRule]:
        return self._cohort_engine.extract_decision_rules()

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert numpy values and containers into JSON-compatible values."""
        if isinstance(value, dict):
            return {
                str(key): ModelExplainer._json_safe(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [ModelExplainer._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return ModelExplainer._json_safe(value.tolist())
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def _clear_cache(self) -> None:
        """Clear cached state to free memory."""
        for name in (
            "_trajectory_metrics",
            "_feature_profiles",
            "_cohorts",
            "_decision_rules",
        ):
            self.__dict__.pop(name, None)
        self._importance_scores.clear()
        self._geographic_aggregation = None

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_config(self) -> PathfinderConfig | None:
        """Return current configuration."""
        return self.config.copy() if self.config else None

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters. Use with caution (clears cache)."""
        self.config.update(kwargs)
        self._clear_cache()

    def __repr__(self) -> str:
        """String representation."""
        n_samples, n_features = self.x_train.shape
        return (
            f"ModelExplainer(n_samples={n_samples}, n_features={n_features}, "
            f"n_trees={len(self.booster.get_dump())})"
        )
