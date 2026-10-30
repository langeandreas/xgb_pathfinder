"""
SHAP engine for xgb-pathfinder.

Computes SHAP values and derives impact metrics:
- mean_abs_shap: Average absolute SHAP value
- high_value_effect: SHAP effect for high feature values
- low_value_effect: SHAP effect for low feature values
- impact_magnitude: Combined effect magnitude
"""

import warnings

import numpy as np
import shap
import xgboost as xgb

from ..config import DEFAULT_CONFIG, SHAP_SAMPLE_SIZE_MAX, TAIL_PERCENTILE
from ..types import DecisionImpactProfile


class ShapEngine:
    """
    Compute SHAP values and decision impact metrics for features.

    Parameters
    ----------
    booster : xgboost.Booster
        Fitted XGBoost booster.
    x_train : np.ndarray
        Training feature matrix.
    feature_names : list[str]
        Human-readable feature names.
    config : dict, optional
        Configuration dict (uses DEFAULT_CONFIG if not provided).
    """

    def __init__(
        self,
        booster: xgb.Booster,
        x_train: np.ndarray,
        feature_names: list[str],
        config: dict | None = None,
    ):
        """Initialize SHAP engine."""
        self.booster = booster
        self.x_train = np.asarray(x_train)
        self.feature_names = feature_names
        self.config = config or DEFAULT_CONFIG
        self._shap_values = None  # Cache
        self._explainer = None  # TreeExplainer cache

    def compute_shap_values(
        self,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        Compute SHAP values using TreeExplainer.

        Parameters
        ----------
        force_recompute : bool
            If True, recompute even if cached.

        Returns
        -------
        np.ndarray
            SHAP values, shape (n_samples, n_features).

        Notes
        -----
        May sample data for efficiency if n_samples > SHAP_SAMPLE_SIZE_MAX.
        """
        if self._shap_values is not None and not force_recompute:
            return self._shap_values

        # Sample data if needed
        x_for_shap = self.x_train
        if len(self.x_train) > SHAP_SAMPLE_SIZE_MAX:
            sample_indices = np.random.choice(
                len(self.x_train),
                size=SHAP_SAMPLE_SIZE_MAX,
                replace=False,
            )
            x_for_shap = self.x_train[sample_indices]
            warnings.warn(
                f"SHAP computation: sampled {SHAP_SAMPLE_SIZE_MAX} from "
                f"{len(self.x_train)} samples for efficiency."
            )

        # Compute SHAP values
        explainer = shap.TreeExplainer(self.booster)
        shap_values = explainer.shap_values(x_for_shap)

        # Handle different output formats
        if isinstance(shap_values, list):
            # Multi-class: take first class
            shap_values = shap_values[0]

        self._shap_values = np.asarray(shap_values)
        return self._shap_values

    def compute_impact_profiles(self) -> dict[str, DecisionImpactProfile]:
        """
        Compute decision impact profiles for all features.

        Returns
        -------
        dict[str, DecisionImpactProfile]
            Maps feature name -> impact profile with SHAP-based metrics.

        Examples
        --------
        >>> profiles = engine.compute_impact_profiles()
        >>> for feat_name, profile in profiles.items():
        ...     print(f"{feat_name}: total_swing={profile['total_swing']:.4f}")
        """
        shap_values = self.compute_shap_values()

        profiles = {}

        for feat_idx in range(shap_values.shape[1]):
            feat_name = (
                self.feature_names[feat_idx]
                if feat_idx < len(self.feature_names)
                else f"feature_{feat_idx}"
            )
            feat_shap = shap_values[:, feat_idx]

            # Compute metrics
            mean_abs_shap = float(np.mean(np.abs(feat_shap)))

            # Tail effects: top/bottom 30% of feature values
            feature_values = self.x_train[:, feat_idx]
            high_percentile = np.percentile(feature_values, 100 - TAIL_PERCENTILE * 100)
            low_percentile = np.percentile(feature_values, TAIL_PERCENTILE * 100)

            high_value_mask = feature_values >= high_percentile
            low_value_mask = feature_values <= low_percentile

            high_value_effect = (
                float(np.mean(np.abs(feat_shap[high_value_mask])))
                if high_value_mask.sum() > 0
                else 0.0
            )
            low_value_effect = (
                float(np.mean(np.abs(feat_shap[low_value_mask])))
                if low_value_mask.sum() > 0
                else 0.0
            )

            total_swing = float(high_value_effect + low_value_effect)

            # Generate interpretation
            interpretation = self._generate_interpretation(
                mean_abs_shap, high_value_effect, low_value_effect
            )

            profile: DecisionImpactProfile = {
                "mean_abs_shap": float(mean_abs_shap),
                "high_value_effect": float(high_value_effect),
                "low_value_effect": float(low_value_effect),
                "total_swing": float(total_swing),
                "interpretation": interpretation,
            }

            profiles[feat_name] = profile

        return profiles

    def _generate_interpretation(
        self,
        mean_abs_shap: float,
        high_value_effect: float,
        low_value_effect: float,
    ) -> str:
        """Generate human-readable interpretation of impact profile."""
        if mean_abs_shap < 0.01:
            return "Minimal impact on predictions"

        if abs(high_value_effect - low_value_effect) < 0.1 * max(
            high_value_effect, low_value_effect
        ):
            return "Symmetric impact (similar effect for high and low values)"

        if high_value_effect > low_value_effect:
            return "Positive correlation: high values increase prediction magnitude"
        else:
            return "Negative correlation: low values increase prediction magnitude"

    def get_feature_importance_ranking(
        self,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank features by mean absolute SHAP value.

        Parameters
        ----------
        top_k : int, optional
            Return only top K features. If None, returns all.

        Returns
        -------
        list[tuple[str, float]]
            list of (feature_name, mean_abs_shap) sorted descending.
        """
        profiles = self.compute_impact_profiles()

        ranking = [
            (feat_name, profile["mean_abs_shap"])
            for feat_name, profile in profiles.items()
        ]

        ranking.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranking = ranking[:top_k]

        return ranking

    def get_impact_magnitude(self) -> dict[str, float]:
        """
        Get impact magnitude (tail effects) for all features.

        Returns
        -------
        dict[str, float]
            Maps feature name -> impact_magnitude (high_effect + low_effect).
        """
        profiles = self.compute_impact_profiles()
        return {
            feat_name: profile["total_swing"] for feat_name, profile in profiles.items()
        }
