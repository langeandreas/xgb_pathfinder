"""
xgb-pathfinder: Generalized XGBoost model explanation package.

Provides trajectory metrics, cohort analysis, and geographic visualization
of xgboost model decisions and impact.

Main API:
---------
from xgb_pathfinder import ModelExplainer

explainer = ModelExplainer(booster, X_train, y_train, feature_names)
trajectories = explainer.compute_trajectory_metrics()
cohorts = explainer.extract_cohorts(min_support=20)
scores = explainer.compute_importance_scores()  # CIS + SDS for all features
region = explainer.extract_region_cohorts("Colombo", mapping_path="[file/path].csv")
data = explainer.to_dict()
"""

from .analysis import (
    compute_importance_scores,
    load_feature_labels,
    plot_importance_scores,
)
from .config import DEFAULT_CONFIG
from .explainer import ModelExplainer
from .geographic import (
    aggregate_cohorts_by_region,
    build_cohort_membership,
    extract_region_cohorts,
    load_admin_geodata,
    load_household_admin_mapping,
    plot_cohort_geography,
    plot_region_cohorts,
    visualise_cohort_map,
    visualise_region_decision_paths,
)
from .types import (
    CohortRecord,
    DecisionRule,
    FeatureProfile,
    FeatureTrajectory,
    ModelExplainerState,
    PathfinderConfig,
    RegionImpactMetric,
    TrajectoryMetric,
)

__version__ = "0.1.0"
__author__ = "xgb-pathfinder Contributors"

__all__ = [
    "DEFAULT_CONFIG",
    "CohortRecord",
    "DecisionRule",
    "FeatureProfile",
    "FeatureTrajectory",
    "ModelExplainer",
    "ModelExplainerState",
    "PathfinderConfig",
    "RegionImpactMetric",
    "TrajectoryMetric",
    "aggregate_cohorts_by_region",
    "build_cohort_membership",
    "compute_importance_scores",
    "extract_region_cohorts",
    "load_admin_geodata",
    "load_feature_labels",
    "load_household_admin_mapping",
    "plot_cohort_geography",
    "plot_importance_scores",
    "plot_region_cohorts",
    "visualise_cohort_map",
    "visualise_region_decision_paths",
]
