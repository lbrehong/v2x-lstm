"""
Tests for scripts/prepare_data.py - cross-RAT GPS matching.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_RATS
from scripts.prepare_data import match_data, secondary_csvs


def _write_trimmed(tmp_path, n, with_dsrc=False):
    coords = {
        "tx_latitude": 43.56 + np.arange(n) * 1e-4,
        "tx_longitude": 1.466 + np.arange(n) * 1e-4,
    }
    pd.DataFrame({
        "tx_seq_num": range(n), "tx_timestamp_ms": np.arange(n) * 2000.0, **coords,
        "latency_ms": 30.0, "sinr": 10.0, "rsrp": -100.0, "pdr": 0.8,
    }).to_csv(tmp_path / "trim_5g.csv", index=False)
    pd.DataFrame({
        "tx_seq_num": range(n), "tx_timestamp_ms": np.arange(n) * 100.0, **coords, "latency_ms": 20.0,
    }).to_csv(tmp_path / "trim_pc5.csv", index=False)
    if with_dsrc:
        pd.DataFrame({
            "tx_seq_num": range(n), "tx_timestamp_ms": np.arange(n) * 20.0, **coords,
            "latency_ms": 5.0, "rsrp_1": -80.0, "rsrp_2": -82.0,
        }).to_csv(tmp_path / "trim_dsrc.csv", index=False)


def test_match_data_pairs_same_place_and_nearest_time(tmp_path):
    """PC5 from another pass at the same place is ignored; beyond the time window there is no match."""
    from scripts.prepare_data import MATCH_TIME_TOLERANCE_MS

    t_a, t_b = 1_000_000.0, 1_100_000.0
    pd.DataFrame({
        "tx_seq_num": [0, 1], "tx_timestamp_ms": [t_a, t_b],
        "tx_latitude": [43.5600, 43.5700], "tx_longitude": [1.4660, 1.4760],
        "latency_ms": [30.0, 31.0], "sinr": 10.0, "rsrp": -100.0, "pdr": 1.0,
    }).to_csv(tmp_path / "trim_5g.csv", index=False)
    pd.DataFrame({
        "tx_seq_num": [0, 1, 2],
        # Point A: an earlier pass 50 min before (listed first), and one 5 s after the ping.
        # Point B: only a measurement well outside the time window.
        "tx_timestamp_ms": [t_a - 3_000_000, t_a + 5_000, t_b + MATCH_TIME_TOLERANCE_MS + 60_000],
        "tx_latitude": [43.5600, 43.5600, 43.5700], "tx_longitude": [1.4660, 1.4660, 1.4760],
        "latency_ms": [99.0, 11.0, 12.0], "pdr": [0.2, 0.95, 0.9],
    }).to_csv(tmp_path / "trim_pc5.csv", index=False)

    match_data(str(tmp_path))

    matched = pd.read_csv(tmp_path / "matched_pc5.csv")
    assert len(matched) == 1
    assert matched["tx_latitude"].iloc[0] == 43.5600
    assert matched["latency_ms"].iloc[0] == 11.0 and matched["pdr"].iloc[0] == 0.95


def test_secondary_csvs_follow_rat_set():
    assert secondary_csvs() == ["trim_pc5.csv"]
    assert secondary_csvs(ALL_RATS) == ["trim_dsrc.csv", "trim_pc5.csv"]


def test_match_data_default_rats_ignore_dsrc(tmp_path):
    _write_trimmed(tmp_path, 10, with_dsrc=True)

    match_data(str(tmp_path))

    assert (tmp_path / "matched_pc5.csv").exists()
    assert not (tmp_path / "matched_dsrc.csv").exists()


def test_match_data_keeps_trimmed_pdr_and_skips_missing_rats(tmp_path):
    n = 30
    _write_trimmed(tmp_path, n)

    # Requesting DSRC on a dataset without trim_dsrc.csv just skips it
    match_data(str(tmp_path), rats=ALL_RATS)

    matched_5g = pd.read_csv(tmp_path / "matched_5g.csv")
    assert len(matched_5g) == n
    assert (matched_5g["pdr"] == 0.8).all()
    assert "pdr" in pd.read_csv(tmp_path / "matched_pc5.csv").columns
    assert not (tmp_path / "matched_dsrc.csv").exists()
