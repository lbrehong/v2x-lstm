"""
Raw log ingestion into split-orient df_*.json files, like cohda's.

Port of cohda's dataset/ingest_json-new.ipynb, so both projects trim the same
raw datasets into the same JSON files:

- datalake._5G_UE.json          -> df_radio.json  (5G radio metrics per UE)
- datalake._5G_RUTX_iperf.json  -> df_iperf.json  (iperf intervals)
- CAM_*.jsonl                   -> df_rtk.json    (RTK positions from CAMs)
- datalake._5G_RUTX_ping.json   -> df_ping.json   (5G ping latency + RTK GPS)
- logs_pc5_<rsu>_*.log          -> df_pc5.json    (PC5 packets + RTK GPS)

All outputs use local time (UTC+1). CAM and PC5 epochs are UTC and are shifted
by LOCAL_TIME_OFFSET_S; datalake $date values are already local time (the UE
records' own 'utc' field is one hour earlier) and are kept as is.

Deliberate differences from the notebook:
- df_ping.json gains a trailing packet_loss column (percent of the ping's
  probes lost), used for 5G PDR. Cohda selects columns by name and ignores it.
- Ping and PC5 positions are matched per row on each record's own time. The
  notebook shifted some pings by +-1h depending on how far the RTK log was,
  and wrote PC5 positions through index labels shared by the three RSU logs,
  which mixed up rows of different RSUs.
- DSRC logs are not ingested.

A raw folder with tour subfolders (e.g. Tour1/, Tour2/) is ingested per tour,
then merged into top-level files like the notebook's first cell.

Usage:
    python -m scripts.ingest_logs --input /path/to/raw_dataset --output /path/to/output
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import re
from io import StringIO

import numpy as np
import pandas as pd

LOCAL_TIME_OFFSET_S = 3600    # CAM and PC5 epochs are UTC; outputs are local time (UTC+1)
GPS_MATCH_TOLERANCE_S = 9     # Max time gap to copy an RTK position onto a row
PC5_RSU_ORDER = ["0cb8", "01e0", "01f0"]  # Concatenation order used by cohda

RADIO_FILE = "datalake._5G_UE.json"
IPERF_FILE = "datalake._5G_RUTX_iperf.json"
PING_FILE = "datalake._5G_RUTX_ping.json"
CAM_PATTERN = "CAM_*.jsonl"
PC5_PATTERN = "logs_pc5_*.log"
RAW_PATTERNS = (RADIO_FILE, IPERF_FILE, PING_FILE, CAM_PATTERN, PC5_PATTERN)

RADIO_CELL_FIELDS = [
    "dl_bitrate", "ul_bitrate", "dl_tx", "ul_tx", "dl_err", "ul_err",
    "dl_retx", "ul_retx", "ul_path_loss", "dl_mcs", "pusch_snr",
]


# ---------------------------------------------------------------------------
# Per-source ingestion
# ---------------------------------------------------------------------------

def ingest_radio(path):
    """Extract per-UE radio metrics from the 5G core's ue_get export (local time)."""
    with open(path) as f:
        data_list = json.load(f)

    radio_data = []
    for data in data_list:
        try:
            timestamp = pd.to_datetime(data["timestamp"]["$date"])
            for ue in data["ue_list"]:
                cell = ue["cells"][0]
                row = {"timestamp": timestamp, "ip": ue["bearers"][1]["ip"]}
                row.update({field: cell[field] for field in RADIO_CELL_FIELDS})
                radio_data.append(row)
        except Exception:
            # As in the notebook: a malformed UE drops the rest of its entry,
            # UEs already appended from that entry are kept
            continue

    return pd.DataFrame(radio_data)


def ingest_iperf(path):
    """Extract one row per iperf reporting interval."""
    with open(path) as f:
        data_list = json.load(f)

    rows = []
    for data in data_list:
        try:
            ip = data.get("ip", "unknown")
            for interval in data["iperf"]["intervals"]:
                s = interval["sum"]
                rows.append({
                    "start_time": s["start"],
                    "end_time": s["end"],
                    "bitrate_mbps": s["bits_per_second"] / 1e6,
                    "jitter_ms": s.get("jitter_ms"),
                    "lost_packets": s.get("lost_packets"),
                    "packets": s.get("packets"),
                    "ip": ip,
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


def _parse_percent(value):
    """'20%' -> 20.0; anything unparseable -> NaN."""
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return np.nan


def ingest_ping(path):
    """Extract 5G pings that report an average latency (local time)."""
    with open(path) as f:
        data_list = json.load(f)

    ping_data = []
    for data in data_list:
        try:
            ip = data.get("ip", "unknown")
            if "ping" in data and "avg_latency" in data["ping"]:
                latency = float(data["ping"]["avg_latency"].split()[0])
                ping_data.append({
                    "timestamp": pd.to_datetime(data["timestamp"]["$date"]),
                    "test_type": "ping",
                    "latency_ms": latency,
                    "ip": ip,
                    "packet_loss": _parse_percent(data["ping"].get("packet_loss")),
                })
        except Exception:
            continue

    return pd.DataFrame(ping_data)


def ingest_rtk(path):
    """Extract RTK positions from a CAM message log, in local time, sorted by time."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                ts = data.get("timestamp")
                if isinstance(ts, dict) and "$date" in ts:
                    ts = pd.to_datetime(ts["$date"])
                elif isinstance(ts, str):
                    ts = pd.to_datetime(ts)
                elif isinstance(ts, int):
                    ts = pd.to_datetime(ts, unit="ms")
                else:
                    ts = pd.NaT

                position = data.get("message", {}).get("basic_container", {}) \
                    .get("reference_position_with_confidence", {})
                lat = position.get("latitude")
                lon = position.get("longitude")

                records.append({
                    "message_type": data.get("message_type"),
                    "source_uuid": data.get("source_uuid"),
                    "timestamp": ts + pd.Timedelta(seconds=LOCAL_TIME_OFFSET_S),
                    "latitude": lat / 10_000_000 if lat is not None else None,
                    "longitude": lon / 10_000_000 if lon is not None else None,
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        # Default (quicksort) like the notebook: it decides the order of duplicate timestamps
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _is_text(column):
    # pandas 2 reads text as object, pandas 3 as str
    return column.dtype == object or pd.api.types.is_string_dtype(column.dtype)


def parse_pc5_log(path):
    """
    Parse one Cohda PC5 receiver log.

    The column header is the line starting with 'rx_timestamp'; data lines
    start with a 15+ digit rx timestamp (statistics lines are skipped).
    Adds 'timestamp' in local epoch seconds.
    """
    with open(path) as f:
        lines = f.readlines()

    columns, data_start = None, None
    for i, line in enumerate(lines):
        if line.strip().startswith("rx_timestamp"):
            columns = [col.strip() for col in line.strip().split(",")]
            data_start = i + 1
            break
    if columns is None:
        raise ValueError(f"No column header starting with 'rx_timestamp' in {path}")

    data_lines = [line.strip() for line in lines[data_start:]
                  if line.strip() and re.match(r"^\d{15,}", line.strip())]
    if not data_lines:
        return pd.DataFrame(columns=columns + ["timestamp"])

    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, names=columns, index_col=False)
    df = df.apply(lambda col: col.str.strip() if _is_text(col) else col)
    df["latency (ms)"] = pd.to_numeric(df["latency (ms)"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["rx_timestamp (us)"], errors="coerce") / 1e6 + LOCAL_TIME_OFFSET_S
    return df


def _rsu_id(path):
    return os.path.basename(path).split("_")[2]


def ingest_pc5(paths):
    """Parse PC5 logs of several RSUs, tagged with their RSU Id, in cohda's order."""
    def order(path):
        rsu = _rsu_id(path)
        rank = PC5_RSU_ORDER.index(rsu) if rsu in PC5_RSU_ORDER else len(PC5_RSU_ORDER)
        return rank, rsu

    frames = []
    for path in sorted(paths, key=order):
        df = parse_pc5_log(path)
        df["Id"] = _rsu_id(path)
        frames.append(df)

    # Each log keeps its own 0..n-1 index, as in cohda's files
    return pd.concat(frames)


# ---------------------------------------------------------------------------
# GPS matching
# ---------------------------------------------------------------------------

def epoch_seconds(timestamps):
    """Datetime-like Series -> float epoch seconds (NaN for NaT), whatever its unit or timezone."""
    ts = pd.to_datetime(pd.Series(timestamps))
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    seconds = ns / 1e9
    seconds[ts.isna().to_numpy()] = np.nan
    return pd.Series(seconds, index=ts.index)


def _nearest_rtk(rtk_epoch, ts):
    """
    Nearest RTK sample for each timestamp.

    Ties go to the earliest RTK row, like Series.idxmin in the notebook.

    Returns:
        Tuple (positions, gaps): RTK row positions (-1 when unavailable) and
        signed gaps rtk_time - ts in seconds (NaN when unavailable).
    """
    positions = np.full(len(ts), -1)
    gaps = np.full(len(ts), np.nan)
    valid = np.flatnonzero(np.isfinite(rtk_epoch))
    finite = np.isfinite(ts)
    if valid.size == 0 or not finite.any():
        return positions, gaps

    order = valid[np.argsort(rtk_epoch[valid], kind="stable")]
    sorted_epoch = rtk_epoch[order]
    t = ts[finite]

    right = np.minimum(np.searchsorted(sorted_epoch, t, side="left"), len(order) - 1)
    left = np.maximum(right - 1, 0)
    # Among equal RTK times, the stable sort puts the earliest row first
    left = np.searchsorted(sorted_epoch, sorted_epoch[left], side="left")

    d_left = np.abs(sorted_epoch[left] - t)
    d_right = np.abs(sorted_epoch[right] - t)
    take_right = (d_right < d_left) | ((d_right == d_left) & (order[right] < order[left]))
    pick = np.where(take_right, right, left)

    positions[finite] = order[pick]
    gaps[finite] = sorted_epoch[pick] - t
    return positions, gaps


def attach_rtk_gps(df, ts_epoch_s, rtk):
    """
    Add Latitude/Longitude from the nearest RTK sample.

    Args:
        df: Rows to locate (ping or PC5).
        ts_epoch_s: Row timestamps in local epoch seconds, one per row of df.
        rtk: df_rtk frame (timestamp, latitude, longitude), or None.

    Returns:
        Copy of df with 'timestamp' set to epoch seconds and Latitude/Longitude
        added (NaN when no RTK sample is within GPS_MATCH_TOLERANCE_S).
    """
    ts = np.asarray(ts_epoch_s, dtype=float)
    latitude = np.full(len(ts), np.nan)
    longitude = np.full(len(ts), np.nan)

    if rtk is not None and not rtk.empty:
        positions, gaps = _nearest_rtk(epoch_seconds(rtk["timestamp"]).to_numpy(), ts)
        matched = np.abs(np.nan_to_num(gaps, nan=np.inf)) <= GPS_MATCH_TOLERANCE_S
        latitude[matched] = pd.to_numeric(rtk["latitude"]).to_numpy(dtype=float)[positions[matched]]
        longitude[matched] = pd.to_numeric(rtk["longitude"]).to_numpy(dtype=float)[positions[matched]]

    out = df.copy()
    out["timestamp"] = ts
    out["Latitude"] = latitude
    out["Longitude"] = longitude
    return out


# ---------------------------------------------------------------------------
# Folders and datasets
# ---------------------------------------------------------------------------

def _find_files(raw_dir, pattern):
    return sorted(glob.glob(os.path.join(raw_dir, pattern)))


def _find_one(raw_dir, pattern):
    matches = _find_files(raw_dir, pattern)
    if len(matches) > 1:
        print(f"  Warning: several files match {pattern} in {raw_dir}, using {os.path.basename(matches[0])}")
    return matches[0] if matches else None


def has_raw_inputs(raw_dir):
    """True if raw_dir directly contains any file ingest_folder can read."""
    return any(_find_files(raw_dir, pattern) for pattern in RAW_PATTERNS)


def _write_df_json(df, path):
    # Epoch timestamps, as in cohda's files (pandas is moving its default to ISO)
    df.to_json(path, orient="split", compression="infer", date_format="epoch")
    print(f"  {os.path.basename(path)}: {len(df)} rows")


def ingest_folder(raw_dir, out_dir):
    """
    Ingest one raw folder (one tour) into df_*.json files.

    Missing sources are skipped. Ping and PC5 positions need the CAM log;
    without it they are written with empty Latitude/Longitude.

    Returns:
        List of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def write(name, df):
        path = os.path.join(out_dir, name)
        _write_df_json(df, path)
        written.append(path)

    radio_file = _find_one(raw_dir, RADIO_FILE)
    if radio_file:
        write("df_radio.json", ingest_radio(radio_file))

    iperf_file = _find_one(raw_dir, IPERF_FILE)
    if iperf_file:
        write("df_iperf.json", ingest_iperf(iperf_file))

    rtk = None
    cam_file = _find_one(raw_dir, CAM_PATTERN)
    if cam_file:
        rtk = ingest_rtk(cam_file)
        write("df_rtk.json", rtk)
    else:
        print(f"  Warning: no {CAM_PATTERN} in {raw_dir}, ping/PC5 rows get no GPS")

    ping_file = _find_one(raw_dir, PING_FILE)
    if ping_file:
        ping = ingest_ping(ping_file)
        if not ping.empty:
            ping = attach_rtk_gps(ping, epoch_seconds(ping["timestamp"]).to_numpy(), rtk)
            # Keep cohda's column order; packet_loss is an addition at the end
            ping = ping[[c for c in ping.columns if c != "packet_loss"] + ["packet_loss"]]
        write("df_ping.json", ping)

    pc5_files = _find_files(raw_dir, PC5_PATTERN)
    if pc5_files:
        pc5 = ingest_pc5(pc5_files)
        pc5 = attach_rtk_gps(pc5, pc5["timestamp"].to_numpy(), rtk)
        write("df_pc5.json", pc5)

    return written


def merge_tours(tour_dirs, out_dir):
    """Concatenate every tour's df_*.json into out_dir, like the notebook's first cell."""
    merged = {}
    for tour_dir in tour_dirs:
        for path in sorted(glob.glob(os.path.join(tour_dir, "df_*.json"))):
            name = os.path.basename(path)
            df = pd.read_json(path, orient="split", compression="infer")
            merged[name] = pd.concat([merged[name], df], ignore_index=True) if name in merged else df

    os.makedirs(out_dir, exist_ok=True)
    for name, df in merged.items():
        _write_df_json(df, os.path.join(out_dir, name))


def ingest_dataset(raw_root, out_root):
    """
    Ingest a raw dataset folder into df_*.json files.

    If raw_root has subfolders containing raw logs (e.g. Tour1/, Tour2/), each
    is ingested into out_root/<name>/ and the results are merged into out_root.
    Otherwise raw_root itself is ingested into out_root.

    Returns:
        List of output folders written (tours first, then out_root).
    """
    tours = sorted(
        name for name in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, name)) and has_raw_inputs(os.path.join(raw_root, name))
    )

    if tours:
        tour_outs = []
        for name in tours:
            print(f"Ingesting {os.path.join(raw_root, name)}")
            tour_out = os.path.join(out_root, name)
            ingest_folder(os.path.join(raw_root, name), tour_out)
            tour_outs.append(tour_out)
        print(f"Merging {len(tours)} tours into {out_root}")
        merge_tours(tour_outs, out_root)
        return tour_outs + [out_root]

    if has_raw_inputs(raw_root):
        print(f"Ingesting {raw_root}")
        ingest_folder(raw_root, out_root)
        return [out_root]

    raise FileNotFoundError(f"No raw logs found in {raw_root} or its subfolders")


def main():
    parser = argparse.ArgumentParser(description="Ingest raw V2X logs into cohda-compatible df_*.json files")
    parser.add_argument("--input", required=True, help="Raw dataset folder (or folder of tour subfolders)")
    parser.add_argument("--output", required=True, help="Output folder for df_*.json files")
    args = parser.parse_args()
    ingest_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
