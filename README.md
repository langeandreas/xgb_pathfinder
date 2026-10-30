# xgb-pathfinder

A generalized, SHAP-like package for explaining XGBoost tree-based models through trajectory metrics, cohort analysis, and geographic impact visualization. This package has been developed in accordance with the methodology of my master's thesis which is currently under review. 


---

## Overview

`xgb-pathfinder` extracts actionable insights from boosting models by analyzing:

1. **Trajectory Metrics** — How feature importance evolves across training iterations
   - Velocity: Feature gain per iteration
   - Acceleration: Change in velocity (identifies turning points)
   - Feature Confusion Index (FCI): Conditional probability metric
   - Experimental/not included in the thesis: Behavioral Signature: CORE_DRIVER, LATE_LEARNER, OUTLIER_SPECIALIST, etc.

2. **Cohort Analysis** — Population segments defined by decision tree patterns
   - Scope signature: Early-stage conditions defining segment
   - Decider feature: Primary decision-making feature
   - Enriched metrics: Support count, risk rate, decision impact, local SHAP summary

3. **Geographic Cohort Summaries** — Where cohort decision paths are represented
    - Mean or cumulative absolute leaf magnitude across household-cohort memberships
    - Regional sample, membership, and distinct-cohort counts
    - Descriptive path evidence, not causal dependence on a cohort

---

## Installation

### Development Install (from project root)

```bash
cd xgb_pathfinder
pip install -e .
```

### User Install

pyPi install/publish coming soon


---

## Quick Start

For an executable walkthrough, open [usage_demo.ipynb](examples/usage_demo.ipynb),
select the project's Python environment, and run all cells. It uses synthetic
households and boundaries, so no external datasets are required. The notebook
covers training, trajectories, cohorts, CIS/SDS, geographic maps, regional
decision paths, and exports using the current API.
Start the kernel in the notebook's `examples` directory to avoid the package's
root-level `types.py` shadowing the standard library. In VS Code, set
`jupyter.notebookFileRoot` to `${fileDirname}` if your kernel starts at the
workspace root.

### Basic API

```python
from xgb_pathfinder import ModelExplainer

# Initialize with trained XGBoost booster
explainer = ModelExplainer(
    booster=trained_model,
    X_train=X_train,
    y_train=y_train,
    feature_names=feature_names,
)

# Compute feature trajectories
trajectories = explainer.compute_trajectory_metrics()
for feat_name, traj in trajectories.items():
    print(f"{feat_name}: peak_velocity={traj['peak_velocity']:.4f}, "
          f"behavior={traj['behavior_signature']}")

# Extract cohorts (population segments)
cohorts = explainer.extract_cohorts(min_support=20)
for cohort in cohorts.head().itertuples(index=False):
    print(f"Cohort {cohort.cohort_id}: "
          f"{cohort.support_count} samples, "
          f"risk={cohort.risk_rate:.2%}, "
          f"impact={cohort.decision_impact:.3f}")



# Geographic aggregation
regions = explainer.aggregate_by_region(
    geodata_path="admin_boundaries.geojson",
    sample_region_mapping=sample_to_region_mapping,
    impact_metric="mean_member_impact",
)

# Visualize impact
fig = explainer.plot_geographic_impact(
    geodata_path="admin_boundaries.geojson",
    sample_region_mapping=sample_to_region_mapping,
    output_path="impact_map.html",
)

# Export metrics
explainer.to_csv("output_dir/")
explainer.to_json("output.json")
data = explainer.to_dict()
```

### Interpreting Regional Cohort Metrics

A cohort is a post-hoc group of repeated XGBoost decision paths. It is not an
input feature, so regional cohort metrics do **not** say that a prediction
causally depends on a cohort.

- `mean_member_impact` (default): mean absolute leaf value across all
    sample-cohort memberships in a region. Use this to compare typical path
    magnitude while limiting the effect of regional sample size.
- `cumulative_member_impact`: sum of those magnitudes. Use this when both path
    magnitude and the number of memberships should matter; larger regions tend
    to score higher.
- `distinct_cohort_count`: number of different extracted cohorts represented
    in the region. This measures diversity, not prediction magnitude.

Each regional result includes `impact_label`, `impact_description`,
`sample_count`, `membership_count`, `mean_member_impact`, and
`cumulative_member_impact`. Map titles, legends, and captions use the same
metadata automatically.

```python
# Typical path magnitude, comparable across differently sized regions
exp.plot_geographic_impact(geodata_path, mapping)

# Total path magnitude, including regional membership frequency
exp.plot_geographic_impact(
    geodata_path, mapping, impact_metric="cumulative_member_impact"
)

# Representation and coverage maps
exp.plot_geographic_impact(geodata_path, mapping, color_by="sample_count")
exp.plot_geographic_impact(geodata_path, mapping, color_by="membership_count")
exp.plot_geographic_impact(geodata_path, mapping, color_by="cohort_count")

# Interactive map
interactive = exp.plot_geographic_impact(
    geodata_path,
    mapping,
    fill_color="YlGnBu",
    zoom_start=7,
)
interactive.save("regional_cohorts.html")
```

### Feature Importance Scores (CIS / SDS)

Same formulas as the analysis app's Feature Story page, computed for all features:

- **CIS** = mean|SHAP| × (1 + tanh(late acceleration)) × (1 + log1p(mean FCI)), normalized to [0, 1]
- **SDS** = context dependency × decision likelihood × absolute-impact gate × support confidence

```python
from xgb_pathfinder import load_feature_labels

scores = exp.compute_importance_scores()  # one row per feature, sorted by CIS
fig = exp.plot_importance_scores(
    sort_by="cis",  # default: all features, ordered by CIS
    feature_labels=load_feature_labels("data/label_file.csv"),
    output_path="results/cis_sds_features.png",  # optional
)
# Or plot an already-computed score table directly:
from xgb_pathfinder import plot_cis_sds
fig = plot_cis_sds(scores, top_n=25, sort_by="cis")
```

### Cohorts for an Administrative Region

Recalculates cohorts using only the training households in the selected region. `sample_ids` must be household ids.

```python
region = exp.extract_region_cohorts(
    "Region Name",  # region name (case-insensitive) or adm_code
    mapping_path="data/file_name.csv",
    admin_level="adm2",
    order_by="support",  # or "confidence", "decision_impact"
    top_n=10,
)
exp.plot_region_cohorts("Region Name", mapping_path="data/file_name.csv")
```

### Standalone Geographic Functions

Membership construction is shared with `analysis.cohort_engine`, and regional
aggregation is shared with `geographic.impact_aggregator`. These functions are
also available from the package root and `geographic`; the visualization module
re-exports them for convenience rather than implementing a second pipeline.

```python
from xgb_pathfinder import (
    aggregate_cohorts_by_region,
    build_cohort_membership,
    load_admin_geodata,
    load_household_admin_mapping,
    plot_cohort_geography,
    visualise_cohort_map,
    visualise_region_decision_paths,
)

cohorts = exp.extract_cohorts()
membership = build_cohort_membership(cohorts, exp.get_segments(), exp.sample_ids)
mapping = load_household_admin_mapping("data/hh_geo_mapping.csv")
polygons = load_admin_geodata("data/geodata.csv")
regions = aggregate_cohorts_by_region(cohorts, mapping, membership)
fig = visualise_cohort_map(regions, polygons, color_by="mean_member_impact")

# Or aggregate and plot in one step:
fig = plot_cohort_geography(
    cohorts, mapping, membership, polygons,
    impact_metric="cumulative_member_impact",
    output_path="cohort_map.png",
)

region = exp.extract_region_cohorts("Colombo", hh_admin_df=mapping)
fig = visualise_region_decision_paths(region, top_n=5)
```

`build_cohort_membership` accepts rule assignments with `sample_index` and
`assigned_rule_ids` columns. Indices address the supplied sample ID sequence;
without it they are used as IDs. Each sample has at most one membership per
cohort, even when it matches several contributing rules. Unmatched samples
remain in the mapping with an empty list.

`aggregate_cohorts_by_region` requires evaluated membership and returns the
same regional metric dictionary as `ModelExplainer.aggregate_by_region`.
An empty membership mapping returns no regional evidence. Both map functions
accept a GeoDataFrame or a geodata file path and return a Matplotlib figure;
they reject empty regional evidence. Decision-path plots use the current
regional cohort table, including its scope, decider evidence, and household
share. These functions use current impact metrics, not the former
`dominant`/`cohort_share`/`risk_rate` map modes.

---

## Module Structure

### Core API (`explainer.py`)

**`ModelExplainer`** — Main entry point

Public methods:
- `compute_trajectory_metrics()` → dict[feature → trajectory metrics]
- `extract_cohorts(min_support)` → DataFrame with one row per cohort
- `get_feature_profiles()` → dict[feature → profile]
- `get_segments(rules)` → DataFrame with sample-to-cohort mapping
- `aggregate_by_region(geodata, mapping)` → dict[region → impact metrics]
- `plot_geographic_impact(geodata, mapping)` → Map object
- `to_dict() / to_json() / to_csv()` → Export metrics

### Analysis Engines (`analysis/`)

| Module | Responsibility |
|--------|-----------------|
| `tree_engine.py` | Tree structure analysis, rule extraction |
| `trajectory_engine.py` | Velocity, acceleration, FCI, behavioral signatures |
| `shap_engine.py` | SHAP value computation, impact profiles |
| `cohort_engine.py` | Cohort extraction, sample assignment, enrichment |
| `feature_analyzer.py` | Feature profiles, interactions, path statistics |
| `context_analyzer.py` | Early/late positioning, scope→decider graphs |
| `utils.py` | Tree traversal, normalization, statistics |

### Geographic Modules (`geographic/`)

| Module | Responsibility |
|--------|-----------------|
| `impact_aggregator.py` | Aggregate cohorts by region |
| `impact_prioritizer.py` | Rank regions with a configurable heuristic path score |
| `geo_plotter.py` | Choropleth visualization (matplotlib/folium) |
| `mappings.py` | Admin hierarchy & sample→region utilities |
| `visualizations.py` | Dataframe-based loaders, cohort impact maps, and regional decision-path plots |



### Type Definitions (`types.py`)

TypedDict definitions for strong typing:
- `TrajectoryMetric` — Per-iteration metrics
- `FeatureTrajectory` — Complete feature trajectory
- `DecisionRule` — Root-to-leaf path rules
- `CohortRecord` — Aggregated cohort
- `FeatureProfile` — Comprehensive feature analysis
- `RegionImpactMetric` — Geographic aggregation
- `PathfinderConfig` — Configuration parameters

### Configuration (`config.py`)

All thresholds are configurable:
- Acceleration thresholds (early/late phase detection)
- Cohort support & path count minimums
- Geographic aggregation settings
- Top-K export values

---

## Data Models

### FeatureTrajectory
(fictional example)

```python
{
    "feature_name": "age",
    "feature_index": 0,
    "metrics": [
        {
            "iteration": 0,
            "velocity": 0.1234,       # Gain at iteration
            "acceleration": nan,      # ΔGain (NaN for iteration 0)
            "fci": 0.567,            # Feature Confusion Index
            "cumulative_gain": 0.1234,
            "hessian_mean": 0.05,
        },
        # ... more iterations
    ],
    "peak_iteration": 15,
    "peak_velocity": 0.8901,
    "total_gain": 5.432,
}
```

### CohortRecord
(fictional example)

```python
{
    "cohort_id": "cohort_0",
    "scope_signature": "age < 35 AND income < 1000",  # Early conditions
    "decider_feature_name": "education_level",        # Primary late decider
    "decider_feature_index": 5,
    "support_count": 245,          # Samples in cohort
    "risk_rate": 0.62,             # Positive rate
    "mean_confidence": 0.71,       # Model's mean prediction
    "late_decider_rate": 0.73,     # Fraction of paths where decider is late
    "decision_impact": 0.45,       # Combined impact score
    "impact_magnitude_ratio": 1.23, # Relative to global mean
    "leaf_values": [0.2, 0.3, 0.25],  # Leaf values when decider appears
    "top_features_by_shap": [("income", 0.12), ("age", 0.08)],
    "contributing_rules": ["tree_0_path_0", "tree_1_path_2"],
}
```

### RegionImpactMetric
(fictional example)

```python
{
    "region_name": "northern_district",
    "region_id": "district_001",
    "cohort_count": 12,            # Unique cohorts in region
    "total_impact": 5.67,          # Sum of impacts
    "avg_impact_per_cohort": 0.47,
    "avg_impact_per_sample": 0.023,
    "total_support": 245,          # Total samples
    "top_cohorts": [
        {"cohort_id": "cohort_3", "support": 45, "impact": 0.89},
        # ... top 5 cohorts
    ],
}
```

---


## Configuration
(fictional example)

Customize via `config` parameter or environment variables:

```python
config = {
    "acceleration_threshold": 0.1,
    "early_acceleration_threshold": 0.05,
    "late_acceleration_threshold": 0.05,
    "min_cohort_support": 10,
    "min_paths_per_cohort": 2,
    "min_scope_features": 2,  # Set to 1 to include single-feature scopes
    "admin_level": 2,
    "top_k_features": 15,
    "json_indent": 2,
}

explainer = ModelExplainer(
    booster, X_train, y_train,
    feature_names=features,
    config=config,
)
```

# References

The methodology and metrics used in this project are inspired by SHAP (SHapley Additive exPlanations) and XGBoost's feature importance analysis. For more information, refer to:

- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Master's thesis by Andreas Lange (currently in the grading process) It has been rewritten from the thesis' original implementation to provide a framework for analyzing xgboost models and has been designed to be applicable to different data sets.

# AI Usage statement
For the development of this project, AI tools were used to assist in drafting and refining documentation, code comments, and explanations. Additionally, AI tools were employed to generate initial code snippets of basic processing and visualization functions for this general implementation without a specific dataset. All AI-generated content was reviewed and edited by the author to ensure accuracy and relevance. The tools used include ChatGPT and Anthropic Claude.