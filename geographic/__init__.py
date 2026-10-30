"""
Geographic visualization and impact aggregation for xgb-pathfinder.

Modules:
--------
- impact_aggregator: Aggregate cohorts by region
- impact_prioritizer: Rank regions with a configurable heuristic path score
- geo_plotter: Choropleth visualization
- mappings: Sample→region mapping and admin hierarchy utilities
"""

from ..analysis.cohort_engine import build_cohort_membership
from .geo_plotter import GeoPlotter
from .impact_aggregator import ImpactAggregator, aggregate_cohorts_by_region
from .impact_prioritizer import ImpactPrioritizer
from .mappings import AdminHierarchyMapper, SampleToRegionMapper
from .region_cohorts import extract_region_cohorts, plot_region_cohorts
from .visualizations import (
	load_admin_geodata,
	load_household_admin_mapping,
	plot_cohort_geography,
	visualise_cohort_map,
	visualise_region_decision_paths,
)

__all__ = [
	"AdminHierarchyMapper",
	"GeoPlotter",
	"ImpactAggregator",
	"ImpactPrioritizer",
	"SampleToRegionMapper",
	"aggregate_cohorts_by_region",
	"build_cohort_membership",
	"extract_region_cohorts",
	"load_admin_geodata",
	"load_household_admin_mapping",
	"plot_cohort_geography",
	"plot_region_cohorts",
	"visualise_cohort_map",
	"visualise_region_decision_paths",
]
