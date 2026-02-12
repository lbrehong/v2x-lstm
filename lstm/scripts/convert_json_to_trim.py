"""
Convert pre-processed df_*.json DataFrames to trim_*.csv format.

The Saturne/Cohda test platform exports data as pandas JSON files
(orient='split').  This script converts them into the standardised
trim_{5g,pc5,dsrc}.csv format consumed by the rest of the pipeline
(learning, selection, feedback loop).

Supported input layouts
-----------------------
1. **Matched directory** (preferred): contains df_ping.json, df_pc5.json,
   df_dsrc.json that already have Latitude/Longitude columns.
   Optionally also df_gps.json for SINR/RSRP.

2. **Base directory**: contains raw df_ping.json, df_pc5.json, df_dsrc.json
   plus df_gps.json (with SINR/RSRP) and df_radio.json.
   GPS is joined from df_gps.json by nearest timestamp.

Usage:
    python -m scripts.convert_json_to_trim \
        --input /path/to/matched/combined \
        --output /path/to/output_dir

    # With separate GPS/radio data:
    python -m scripts.convert_json_to_trim \
        --input /path/to/base_dir \
        --gps /path/to/df_gps.json \
        --output /path/to/output_dir
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df_json(path: str) -> pd.DataFrame:
    """Load a pandas split-orient JSON file."""
    return pd.read_json(path, orient="split")


def _ts_to_epoch_ms(ts_series: pd.Series) -> pd.Series:
    """Convert a datetime Series to epoch milliseconds (float)."""
    return ts_series.astype(np.int64) / 1e6


def _merge_asof_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_col: str,
    right_col: str,
    columns: list[str],
    tolerance_ms: int = 5000,
) -> pd.DataFrame:
    """Merge *columns* from *right* into *left* by nearest timestamp.

    Both timestamp columns must be in epoch-ms (float/int).
    """
    left = left.sort_values(left_col).reset_index(drop=True)
    right = right.sort_values(right_col).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right[[right_col] + columns].rename(columns={right_col: left_col}),
        on=left_col,
        direction="nearest",
        tolerance=tolerance_ms,
    )
    return merged


# ---------------------------------------------------------------------------
# Per-RAT converters
# ---------------------------------------------------------------------------

def convert_5g(
    input_dir: str,
    gps_path: str | None,
) -> pd.DataFrame | None:
    """Convert df_ping.json (+ optional df_gps.json) -> trim_5g format."""
    ping_path = os.path.join(input_dir, "df_ping.json")
    if not os.path.exists(ping_path):
        print("  df_ping.json not found — skipping 5G")
        return None

    df = _load_df_json(ping_path)

    # Filter to ping-type rows only
    if "test_type" in df.columns:
        df = df[df["test_type"] == "ping"].copy()

    # Timestamp -> epoch ms
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["tx_timestamp_ms"] = _ts_to_epoch_ms(df["timestamp"])
    else:
        df["tx_timestamp_ms"] = df["timestamp"].astype(float)

    # Sequential numbering
    df["tx_seq_num"] = range(len(df))

    # GPS coordinates
    has_gps = "Latitude" in df.columns and df["Latitude"].notna().any()
    if has_gps:
        df["tx_latitude"] = df["Latitude"]
        df["tx_longitude"] = df["Longitude"]
    else:
        df["tx_latitude"] = np.nan
        df["tx_longitude"] = np.nan

    # SINR / RSRP — try df_gps.json
    gps_file = gps_path or os.path.join(input_dir, "df_gps.json")
    if os.path.exists(gps_file):
        gps = _load_df_json(gps_file)
        if pd.api.types.is_datetime64_any_dtype(gps["timestamp"]):
            gps["_ts_ms"] = _ts_to_epoch_ms(gps["timestamp"])
        else:
            gps["_ts_ms"] = gps["timestamp"].astype(float)

        cols_to_merge = [c for c in ["sinr", "rsrp"] if c in gps.columns]
        if cols_to_merge:
            df = _merge_asof_nearest(
                df, gps, "tx_timestamp_ms", "_ts_ms", cols_to_merge,
            )

        # Fill GPS from df_gps if ping had no GPS
        if not has_gps and "GPS.latitude" in gps.columns:
            gps_loc = gps[gps["GPS.latitude"] != 0].copy()
            if len(gps_loc):
                df = _merge_asof_nearest(
                    df, gps_loc, "tx_timestamp_ms", "_ts_ms",
                    ["GPS.latitude", "GPS.longitude"],
                )
                df["tx_latitude"] = df["tx_latitude"].fillna(df.get("GPS.latitude"))
                df["tx_longitude"] = df["tx_longitude"].fillna(df.get("GPS.longitude"))
                df.drop(columns=["GPS.latitude", "GPS.longitude"], inplace=True, errors="ignore")

    # Ensure sinr/rsrp columns exist even without df_gps.json
    if "sinr" not in df.columns:
        df["sinr"] = np.nan
    if "rsrp" not in df.columns:
        df["rsrp"] = np.nan

    out = df[["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
              "latency_ms", "sinr", "rsrp"]].copy()
    out.dropna(subset=["latency_ms"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def convert_pc5(input_dir: str) -> pd.DataFrame | None:
    """Convert df_pc5.json -> trim_pc5 format."""
    path = os.path.join(input_dir, "df_pc5.json")
    if not os.path.exists(path):
        print("  df_pc5.json not found — skipping PC5")
        return None

    df = _load_df_json(path)

    # tx_seq_num
    if "tx_seq_num" in df.columns:
        df["tx_seq_num"] = df["tx_seq_num"].astype(int)
    else:
        df["tx_seq_num"] = range(len(df))

    # Timestamp: prefer tx_timestamp (us) column
    if "tx_timestamp (us)" in df.columns:
        df["tx_timestamp_ms"] = df["tx_timestamp (us)"].astype(float) / 1000.0
    elif pd.api.types.is_datetime64_any_dtype(df.get("timestamp")):
        df["tx_timestamp_ms"] = _ts_to_epoch_ms(df["timestamp"])
    else:
        df["tx_timestamp_ms"] = df["timestamp"].astype(float)

    # GPS: prefer matched Latitude/Longitude, fall back to tx_latitude/tx_longitude
    if "Latitude" in df.columns and df["Latitude"].notna().any():
        df["tx_latitude"] = df["Latitude"]
        df["tx_longitude"] = df["Longitude"]
    elif "tx_latitude" in df.columns:
        df["tx_latitude"] = pd.to_numeric(df["tx_latitude"], errors="coerce")
        df["tx_longitude"] = pd.to_numeric(df["tx_longitude"], errors="coerce")
    else:
        df["tx_latitude"] = np.nan
        df["tx_longitude"] = np.nan

    # Latency
    if "latency (ms)" in df.columns:
        df["latency_ms"] = df["latency (ms)"]
    elif "Latency" in df.columns:
        df["latency_ms"] = df["Latency"]

    # Filter zero-latency rows (same logic as trim_pc5)
    df = df[df["latency_ms"] > 0].copy()

    out = df[["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
              "latency_ms"]].copy()
    out.dropna(subset=["latency_ms"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def convert_dsrc(input_dir: str) -> pd.DataFrame | None:
    """Convert df_dsrc.json -> trim_dsrc format."""
    path = os.path.join(input_dir, "df_dsrc.json")
    if not os.path.exists(path):
        print("  df_dsrc.json not found — skipping DSRC")
        return None

    df = _load_df_json(path)

    # Seq num
    if "SeqNum" in df.columns:
        df["tx_seq_num"] = df["SeqNum"].astype(int)
    else:
        df["tx_seq_num"] = range(len(df))

    # Timestamp
    if "TimeStamp(s)" in df.columns:
        ts = df["TimeStamp(s)"]
        if pd.api.types.is_datetime64_any_dtype(ts):
            df["tx_timestamp_ms"] = _ts_to_epoch_ms(ts)
        else:
            df["tx_timestamp_ms"] = ts.astype(float) * 1000.0
    elif pd.api.types.is_datetime64_any_dtype(df.get("timestamp")):
        df["tx_timestamp_ms"] = _ts_to_epoch_ms(df["timestamp"])
    else:
        df["tx_timestamp_ms"] = df["timestamp"].astype(float)

    # GPS
    if "Latitude" in df.columns and df["Latitude"].notna().any():
        df["tx_latitude"] = df["Latitude"]
        df["tx_longitude"] = df["Longitude"]
    else:
        df["tx_latitude"] = np.nan
        df["tx_longitude"] = np.nan

    # Power / RSRP
    df["rsrp_1"] = df["PowerAnt1"].astype(float) if "PowerAnt1" in df.columns else np.nan
    df["rsrp_2"] = df["PowerAnt2"].astype(float) if "PowerAnt2" in df.columns else np.nan

    # Latency: Lat(us) is in microseconds
    if "Lat(us)" in df.columns:
        df["latency_ms"] = df["Lat(us)"].astype(float) / 1000.0
    elif "latency_ms" in df.columns:
        pass  # already good
    else:
        print("  Warning: no latency column found in DSRC data")
        df["latency_ms"] = np.nan

    out = df[["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
              "rsrp_1", "rsrp_2", "latency_ms"]].copy()
    out.dropna(subset=["latency_ms"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_all(input_dir: str, output_dir: str, gps_path: str | None = None):
    """Convert all available df_*.json files to trim_*.csv."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 5G
    df_5g = convert_5g(input_dir, gps_path)
    if df_5g is not None:
        out_path = os.path.join(output_dir, "trim_5g.csv")
        df_5g.to_csv(out_path, index=False)
        print(f"  trim_5g.csv: {len(df_5g)} rows")
        results["5g"] = len(df_5g)

    # PC5
    df_pc5 = convert_pc5(input_dir)
    if df_pc5 is not None:
        out_path = os.path.join(output_dir, "trim_pc5.csv")
        df_pc5.to_csv(out_path, index=False)
        print(f"  trim_pc5.csv: {len(df_pc5)} rows")
        results["pc5"] = len(df_pc5)

    # DSRC
    df_dsrc = convert_dsrc(input_dir)
    if df_dsrc is not None:
        out_path = os.path.join(output_dir, "trim_dsrc.csv")
        df_dsrc.to_csv(out_path, index=False)
        print(f"  trim_dsrc.csv: {len(df_dsrc)} rows")
        results["dsrc"] = len(df_dsrc)

    if not results:
        print("  No df_*.json files found — nothing converted")
    else:
        print(f"\n  Converted {len(results)} RAT(s) -> {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert df_*.json (pandas split-orient) to trim_*.csv"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing df_ping.json, df_pc5.json, etc.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for trim_*.csv (default: same as --input)")
    parser.add_argument("--gps", type=str, default=None,
                        help="Path to df_gps.json for SINR/RSRP (auto-detected if in --input)")
    args = parser.parse_args()

    output_dir = args.output or args.input
    print(f"Converting df_*.json from {args.input}")
    convert_all(args.input, output_dir, gps_path=args.gps)


if __name__ == "__main__":
    main()
