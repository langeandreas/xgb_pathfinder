"""
Analysis engines for xgb-pathfinder.

Modules:
--------
- trajectory_engine: Velocity, acceleration, FCI metrics
- tree_engine: Tree structure analysis and rule extraction
- shap_engine: SHAP value computation and analysis
- cohort_engine: Cohort extraction and enrichment
- feature_analyzer: Feature-level profiles and statistics
- context_analyzer: Decision path context and scope→decider graphs
- utils: Shared utilities (tree traversal, rule evaluation, etc.)
"""

from .cohort_engine import CohortEngine, build_cohort_membership
from .context_analyzer import ContextAnalyzer
from .feature_analyzer import FeatureAnalyzer
from .importance_scores import (
    compute_importance_scores,
    load_feature_labels,
    plot_importance_scores,
)
from .shap_engine import ShapEngine
from .trajectory_engine import TrajectoryEngine
from .tree_engine import TreeEngine

__all__ = [
    "CohortEngine",
    "ContextAnalyzer",
    "FeatureAnalyzer",
    "ShapEngine",
    "TrajectoryEngine",
    "TreeEngine",
    "build_cohort_membership",
    "compute_importance_scores",
    "load_feature_labels",
    "plot_importance_scores",
]
