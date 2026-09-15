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
import io
import json

import numpy as np
import pandas as pd

from config import LATENCY_BOUNDS

# Max gap between a ping and the radio sample joined onto it. Beyond it the ping
# gets no SINR/RSRP (never a stale value carried over from minutes away).
RADIO_MATCH_TOLERANCE_MS = 10_000

# 5G PDR source. False: trim_5g.csv has no pdr column, so matching/training compute a
# rolling count of received pings, the only definition every dataset supports (cohda's
# matched tours have no per-ping packet_loss). True: 1 - packet_loss of each ping.
FIVEG_PDR_FROM_PACKET_LOSS = False

_INT64_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df_json(path: str) -> pd.DataFrame:
    """
    Load a pandas split-orient JSON file.

    Some exports hold integers beyond int64 (e.g. 2^64 counter sentinels in
    cohda's df_radio.json), which pandas' JSON reader rejects. Those values are
    replaced with NaN and the file is parsed again with the same type inference.
    """
    try:
        return pd.read_json(path, orient="split")
    except ValueError as err:
        if "too big" not in str(err):
            raise

    with open(path) as f:
        raw = json.load(f)
    n_fixed = 0

    def clean(value):
        nonlocal n_fixed
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) > _INT64_MAX:
            n_fixed += 1
            return None
        return value

    raw["data"] = [[clean(v) for v in row] for row in raw["data"]]
    print(f"  {os.path.basename(path)}: {n_fixed} out-of-range integer value(s) set to NaN")
    return pd.read_json(io.StringIO(json.dumps(raw)), orient="split")


def _ts_to_epoch_ms(ts_series: pd.Series) -> pd.Series:
    """Convert a datetime Series to epoch milliseconds (float), whatever its unit or timezone."""
    ts = pd.to_datetime(ts_series)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    # read_json may return datetime64[ms] (pandas 3): normalise to ns before casting
    return pd.Series(ts.to_numpy(dtype="datetime64[ns]").astype(np.int64) / 1e6, index=ts.index)


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

def _merge_radio(df: pd.DataFrame, radio: pd.DataFrame) -> pd.DataFrame:
    """Join 5G radio metrics from df_radio.json onto ping rows of the same UE IP.

    Both files carry the datalake time, so each ping takes the nearest radio
    sample of its UE. The UE log has no SINR/RSRP: sinr takes pusch_snr and
    rsrp takes -ul_path_loss as a proxy. Pings without a radio sample within
    RADIO_MATCH_TOLERANCE_MS keep NaN (e.g. when the radio log stops before the
    pings), so they are treated as missing instead of reusing old values.
    """
    df = df.copy()
    df["_order"] = np.arange(len(df))
    df["_ts_ms"] = df["tx_timestamp_ms"].astype(float)

    radio = radio.dropna(subset=["timestamp"]).copy()
    radio["_ts_ms"] = _ts_to_epoch_ms(radio["timestamp"]).astype(float)

    parts = []
    for ip, rows in df.groupby("ip", sort=False):
        rows = rows.sort_values("_ts_ms")
        samples = radio.loc[radio["ip"] == ip, ["_ts_ms", "pusch_snr", "ul_path_loss"]]
        if samples.empty:
            parts.append(rows.assign(pusch_snr=np.nan, ul_path_loss=np.nan))
            continue
        parts.append(pd.merge_asof(
            rows, samples.sort_values("_ts_ms"), on="_ts_ms",
            direction="nearest", tolerance=float(RADIO_MATCH_TOLERANCE_MS),
        ))

    merged = pd.concat(parts).sort_values("_order")
    print(f"  5G radio metrics matched for {merged['pusch_snr'].notna().mean():.1%} of pings "
          f"(within {RADIO_MATCH_TOLERANCE_MS / 1000:.0f} s)")
    merged["sinr"] = merged["pusch_snr"]
    merged["rsrp"] = -merged["ul_path_loss"]
    return merged.drop(columns=["_order", "_ts_ms", "pusch_snr", "ul_path_loss"]).reset_index(drop=True)


def convert_5g(
    input_dir: str,
    gps_path: str | None,
) -> pd.DataFrame | None:
    """Convert df_ping.json (+ df_radio.json or df_gps.json) -> trim_5g format.

    PDR comes from each ping's packet_loss (share of its probes lost) only when
    FIVEG_PDR_FROM_PACKET_LOSS is set; by default no pdr column is written and
    a rolling count of received pings is computed downstream. SINR/RSRP come
    from df_radio.json when present (see _merge_radio), otherwise from df_gps.json.
    """
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

    # PDR from the ping's own probe loss (off by default, see FIVEG_PDR_FROM_PACKET_LOSS)
    if FIVEG_PDR_FROM_PACKET_LOSS and "packet_loss" in df.columns:
        df["pdr"] = 1.0 - pd.to_numeric(df["packet_loss"], errors="coerce") / 100.0

    # SINR / RSRP — prefer df_radio.json
    radio_path = os.path.join(input_dir, "df_radio.json")
    if os.path.exists(radio_path) and "ip" in df.columns:
        df = _merge_radio(df, _load_df_json(radio_path))

    # SINR / RSRP (if still missing) and GPS fallback — try df_gps.json
    gps_file = gps_path or os.path.join(input_dir, "df_gps.json")
    if os.path.exists(gps_file):
        gps = _load_df_json(gps_file)
        if pd.api.types.is_datetime64_any_dtype(gps["timestamp"]):
            gps["_ts_ms"] = _ts_to_epoch_ms(gps["timestamp"])
        else:
            gps["_ts_ms"] = gps["timestamp"].astype(float)

        cols_to_merge = [c for c in ["sinr", "rsrp"] if c in gps.columns and c not in df.columns]
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

    columns = ["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
               "latency_ms", "sinr", "rsrp"]
    if "pdr" in df.columns:
        columns.append("pdr")
    out = df[columns].copy()
    out.dropna(subset=["latency_ms"], inplace=True)
    # Clip multi-second stalls to the 5G latency bound: they stay "very bad" without dominating the loss
    out["latency_ms"] = out["latency_ms"].clip(upper=LATENCY_BOUNDS["5g"][1])
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

    # Timestamp on the 5G pings' clock: ingest_logs' local `timestamp` when present
    # (raw tx epochs are UTC, 1 h off the local ping time); cohda exports have no such
    # column and their tx epoch already shares the ping clock.
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["tx_timestamp_ms"] = _ts_to_epoch_ms(df["timestamp"])
    elif "tx_timestamp (us)" in df.columns:
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

    # Rows without a transmitter position can't be placed or matched. Counting them
    # would also inflate the rolling PDR when a second transmitter logs no GPS
    # (cohda's matched tours), so PDR describes positioned transmissions only.
    n_before = len(df)
    df = df[df["tx_latitude"].notna() & df["tx_longitude"].notna()].copy()
    if len(df) < n_before:
        print(f"  PC5: dropped {n_before - len(df)} of {n_before} rows without a transmitter position")

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

def _has_df_json(path: str) -> bool:
    return any(os.path.exists(os.path.join(path, name)) for name in ("df_ping.json", "df_pc5.json"))


def _subdirs(path: str) -> list[str]:
    return sorted(os.path.join(path, n) for n in os.listdir(path) if os.path.isdir(os.path.join(path, n)))


def find_tour_dirs(root: str, exclude: str | None = None) -> list[str]:
    """
    Tour folders holding df_*.json exports under root.

    Looks at root's subfolders first (e.g. matched/combined -> Tour1, Tour2, ...),
    then one level deeper (e.g. matched -> combined/Tour1, ...). Folders inside
    `exclude` (typically the output folder) are ignored.
    """
    excluded = os.path.abspath(exclude) if exclude else None

    def keep(path):
        path = os.path.abspath(path)
        return not excluded or os.path.commonpath([path, excluded]) != excluded

    tours = [d for d in _subdirs(root) if keep(d) and _has_df_json(d)]
    if not tours:
        tours = [g for c in _subdirs(root) if keep(c) for g in _subdirs(c) if keep(g) and _has_df_json(g)]
    return tours


def convert_tours(root: str, output_dir: str, gps_path: str | None = None) -> list[str]:
    """
    Convert every tour's df_*.json under root, then merge the tours' trim_*.csv.

    Each tour is converted on its own (its own df_gps.json / df_radio.json) into
    output_dir/<tour>/, and the per-RAT files are concatenated in time order into
    output_dir/trim_<rat>.csv. Cohda's own top-level merged df_*.json are not used.

    Returns:
        List of converted tour folders

    Raises:
        FileNotFoundError: If no tour folder with df_*.json is found
    """
    tours = find_tour_dirs(root, exclude=output_dir)
    if not tours:
        raise FileNotFoundError(f"No tour folders with df_ping.json/df_pc5.json under {root}")

    parts: dict[str, list[pd.DataFrame]] = {}
    for tour in tours:
        tour_out = os.path.join(output_dir, os.path.relpath(tour, root).replace(os.sep, "_"))
        print(f"Converting {tour} -> {tour_out}")
        convert_all(tour, tour_out, gps_path)
        for rat in ("5g", "pc5", "dsrc"):
            path = os.path.join(tour_out, f"trim_{rat}.csv")
            if os.path.exists(path):
                parts.setdefault(rat, []).append(pd.read_csv(path))

    print(f"Merging {len(tours)} tours into {output_dir}")
    for rat, frames in parts.items():
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values("tx_timestamp_ms", kind="stable").reset_index(drop=True)
        merged.to_csv(os.path.join(output_dir, f"trim_{rat}.csv"), index=False)
        print(f"  trim_{rat}.csv: {len(merged)} rows from {len(frames)} tour(s)")
    return tours


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
    parser.add_argument("--input", type=str, default=None,
                        help="Directory containing df_ping.json, df_pc5.json, etc.")
    parser.add_argument("--tours", type=str, default=None,
                        help="Folder of tour subfolders with df_*.json (e.g. matched/combined): "
                             "each tour is converted, then merged (default output: <tours>/trimmed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for trim_*.csv (default: same as --input)")
    parser.add_argument("--gps", type=str, default=None,
                        help="Path to df_gps.json for SINR/RSRP (auto-detected if in --input)")
    args = parser.parse_args()

    if bool(args.input) == bool(args.tours):
        parser.error("provide exactly one of --input or --tours")
    if args.tours:
        convert_tours(args.tours, args.output or os.path.join(args.tours, "trimmed"), gps_path=args.gps)
        return

    output_dir = args.output or args.input
    print(f"Converting df_*.json from {args.input}")
    convert_all(args.input, output_dir, gps_path=args.gps)


if __name__ == "__main__":
    main()
