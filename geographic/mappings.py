"""
Mapping utilities for xgb-pathfinder.

Helpers for working with geographic data:
- Loading/validating admin hierarchy mappings
- Sample to region assignment
- Region name standardization
"""

import numpy as np
import pandas as pd


class AdminHierarchyMapper:
    """
    Manage hierarchical administrative regions.

    Supports levels like: country → region → district

    Parameters
    ----------
    hierarchy_data : dict or pd.DataFrame
        Admin hierarchy definition.
        If dict: {level: {id: name}, ...}
        If DataFrame: columns=['admin_0', 'admin_1', 'admin_2', ...]
    """

    def __init__(
        self,
        hierarchy_data: dict | pd.DataFrame,
    ):
        """Initialize mapper."""
        if isinstance(hierarchy_data, pd.DataFrame):
            self.hierarchy_df = hierarchy_data
            self.levels = [
                col for col in hierarchy_data.columns if col.startswith("admin_")
            ]
        else:
            self.hierarchy_dict = hierarchy_data
            self.levels = sorted(hierarchy_data.keys())

    def get_level_names(self, level: int) -> list[str]:
        """
        Get all unique names at a given admin level.

        Parameters
        ----------
        level : int
            Admin level (1, 2, 3, etc.)

        Returns
        -------
        list[str]
            Unique region names at that level.
        """
        if hasattr(self, "hierarchy_df"):
            col_name = f"admin_{level}"
            if col_name in self.hierarchy_df.columns:
                return self.hierarchy_df[col_name].unique().tolist()

        if level in self.hierarchy_dict:
            return list(self.hierarchy_dict[level].values())

        return []

    def standardize_region_name(
        self,
        region_name: str,
        level: int,
    ) -> str | None:
        """
        Handle whitespace and case-sensitivity.

        Parameters
        ----------
        region_name : str
            Input region name.
        level : int
            Admin level.

        Returns
        -------
        str or None
            Standardized name, or None if not found.
        """
        # Get valid names at this level
        valid_names = self.get_level_names(level)

        # Case-insensitive match
        lower_input = region_name.strip().lower()
        if lower_input in valid_names:
            return lower_input

        return None


class SampleToRegionMapper:
    """
    Map samples to geographic regions.

    Parameters
    ----------
    sample_ids : list or np.ndarray
        Sample identifiers (indices or external IDs).
    region_mapping : dict or pd.DataFrame
        Maps sample ID to region name.
        If dict: {sample_id: region_name, ...}
        If DataFrame: index or column must be sample IDs, values are region names.
    """

    def __init__(
        self,
        sample_ids: list | np.ndarray,
        region_mapping: dict | pd.DataFrame,
    ):
        """Initialize mapper."""
        self.sample_ids = np.asarray(sample_ids)

        if isinstance(region_mapping, pd.DataFrame):
            # Assume first column (or index) is sample ID, second column is region
            if (
                region_mapping.index.name == "sample_id"
                or region_mapping.index.name == "index"
            ):
                self.mapping_dict = region_mapping.iloc[:, 0].to_dict()
            else:
                self.mapping_dict = dict(
                    zip(region_mapping.iloc[:, 0], region_mapping.iloc[:, 1])
                )
        else:
            self.mapping_dict = dict(region_mapping)

    def get_region(self, sample_id: int | str) -> str | None:
        """
        Get region for a sample.

        Parameters
        ----------
        sample_id : int or str
            Sample identifier.

        Returns
        -------
        str or None
            Region name, or None if not found.
        """
        return self.mapping_dict.get(sample_id)

    def get_regions_for_samples(
        self,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Get regions for multiple samples.

        Parameters
        ----------
        sample_indices : np.ndarray
            Indices into sample_ids array.

        Returns
        -------
        np.ndarray
            Region names corresponding to samples.
        """
        regions = []
        for idx in sample_indices:
            sample_id = self.sample_ids[idx]
            region = self.mapping_dict.get(key=sample_id, default="unknown")  # type: ignore
            regions.append(region)
        return np.array(regions)

    def get_region_sample_counts(self) -> dict[str, int]:
        """
        Count samples per region.

        Returns
        -------
        dict[str, int]
            Maps region name -> sample count.
        """
        counts = {}
        for region in self.mapping_dict.values():
            counts[region] = counts.get(region, 0) + 1
        return counts


def load_sample_to_region_mapping(
    mapping_path: str,
    sample_id_col: str | None = None,
    region_col: str | None = None,
) -> dict:
    """
    Load sample-to-region mapping from CSV file.

    Parameters
    ----------
    mapping_path : str
        Path to CSV file.
    sample_id_col : str, optional
        Column name for sample IDs. Auto-detected if not specified.
    region_col : str, optional
        Column name for regions. Auto-detected if not specified.

    Returns
    -------
    dict
        Maps sample_id -> region_name.
    """
    df = pd.read_csv(mapping_path)

    # Auto-detect columns if not specified
    if sample_id_col is None:
        sample_id_col = df.columns[0]
    if region_col is None:
        region_col = df.columns[1]

    return dict(zip(df[sample_id_col], df[region_col]))


def validate_sample_region_mapping(
    mapping: dict,
    n_samples: int,
    n_regions: int | None = None,
) -> bool:
    """
    Validate sample-to-region mapping.

    Parameters
    ----------
    mapping : dict
        Sample to region mapping.
    n_samples : int
        Expected number of samples.
    n_regions : int, optional
        Expected number of unique regions (min check).

    Returns
    -------
    bool
        True if mapping is valid.
    """
    # Check coverage
    if len(mapping) < n_samples * 0.9:  # Allow 10% missing
        return False

    # Check region count
    return not (
        n_regions is not None and len(set(mapping.values())) < n_regions * 0.8
    )  # Allow 20% missing
