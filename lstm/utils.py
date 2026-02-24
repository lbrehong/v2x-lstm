"""
Shared utility functions for the RAT prediction project.
"""
import os
import glob
import re
from typing import Optional, List

import pandas as pd

from config import MODEL_DIR
from api_types import NetworkState


def find_files_with_string(directory: str, search_string: str) -> List[str]:
    """
    Find files in a directory matching a substring pattern.

    Args:
        directory: Path to search in
        search_string: Substring to match in filenames

    Returns:
        List of matching filenames (not full paths)
    """
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if search_string in f]
    return matching_files


def get_latest_model(
    model_type: str,
    rat: str,
    model_dir: str = MODEL_DIR,
    include_retrained: bool = False,
) -> Optional[str]:
    """
    Find the most recent model file based on the model type and RAT.

    Args:
        model_type: Type of model ('lstm', 'gru', 'rnn')
        rat: RAT type ('5g', 'pc5', 'dsrc')
        model_dir: Directory containing saved models
        include_retrained: If False (default), skip retrained_* files

    Returns:
        Path to most recent model, or None if not found
    """
    pattern = os.path.join(model_dir, f"*{model_type}_{rat}_*.keras")
    model_files = glob.glob(pattern)

    regex = re.compile(rf"(?:retrained_)?{model_type}_{rat}_(\d+)\.keras")
    files_with_time = []

    for filepath in model_files:
        basename = os.path.basename(filepath)
        if not include_retrained and basename.startswith("retrained_"):
            continue
        match = regex.search(basename)
        if match:
            files_with_time.append((filepath, int(match.group(1))))

    if not files_with_time:
        print("No existing models found.")
        return None

    latest_model = max(files_with_time, key=lambda x: x[1])[0]
    print(f"Found a model: {latest_model}")
    return latest_model


def ensure_dir_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def safe_float(value) -> Optional[float]:
    """
    Safely convert value to float, returning None for invalid values.

    Args:
        value: Value to convert (can be int, float, str, None, NaN)

    Returns:
        Float value, or None if conversion fails or value is None/NaN
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def row_to_network_state(
    row,
    default_lat: float = 0.0,
    default_lon: float = 0.0,
    timestamp_ms: Optional[int] = None,
) -> NetworkState:
    """
    Convert a DataFrame row (or dict-like) to a NetworkState.

    Handles various column naming conventions from different data sources.

    Args:
        row: DataFrame row or dict-like with network measurements
        default_lat: Default latitude when GPS columns are missing
        default_lon: Default longitude when GPS columns are missing
        timestamp_ms: Override timestamp (used by queue simulator)

    Returns:
        NetworkState instance
    """
    # Resolve timestamp
    if timestamp_ms is None:
        timestamp_ms = 0
        for col in ["timestamp_ms", "timestamp", "time_ms"]:
            if col in row and pd.notna(row[col]):
                timestamp_ms = int(row[col])
                break

    # GPS coordinates
    lat = row.get("tx_latitude", row.get("latitude", default_lat))
    lon = row.get("tx_longitude", row.get("longitude", default_lon))

    return NetworkState(
        timestamp_ms=timestamp_ms,
        latitude=float(lat),
        longitude=float(lon),
        # DSRC measurements
        dsrc_latency_ms=safe_float(row.get("latency_ms_dsrc", row.get("dsrc_latency_ms"))),
        dsrc_pdr=safe_float(row.get("pdr_dsrc", row.get("dsrc_pdr"))),
        dsrc_rsrp_1=safe_float(row.get("rsrp_1", row.get("dsrc_rsrp_1"))),
        dsrc_rsrp_2=safe_float(row.get("rsrp_2", row.get("dsrc_rsrp_2"))),
        # PC5 measurements
        pc5_latency_ms=safe_float(row.get("latency_ms_pc5", row.get("pc5_latency_ms"))),
        pc5_pdr=safe_float(row.get("pdr_pc5", row.get("pc5_pdr"))),
        # 5G measurements
        fiveg_latency_ms=safe_float(row.get("latency_ms_5g", row.get("fiveg_latency_ms", row.get("latency_ms")))),
        fiveg_pdr=safe_float(row.get("pdr_5g", row.get("fiveg_pdr", row.get("pdr")))),
        fiveg_sinr=safe_float(row.get("sinr", row.get("fiveg_sinr"))),
        fiveg_rsrp=safe_float(row.get("rsrp", row.get("fiveg_rsrp"))),
    )
