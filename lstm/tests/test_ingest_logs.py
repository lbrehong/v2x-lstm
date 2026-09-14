"""
Tests for scripts/ingest_logs.py - raw log ingestion into cohda-compatible df_*.json files.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_logs import (
    GPS_MATCH_TOLERANCE_S, LOCAL_TIME_OFFSET_S, RADIO_CELL_FIELDS,
    attach_rtk_gps, epoch_seconds, ingest_dataset, ingest_iperf,
    ingest_pc5, ingest_ping, ingest_radio, ingest_rtk, parse_pc5_log,
)

PC5_COLUMNS = [
    "rx_timestamp (us)", "rx_receiver_fix_mode", "rx_latitude", "rx_longitude", "rx_altitude (m)",
    "rx_quantity_of_SV_used", "rx_Semi_Major_Axis_Accuracy", "rx_heading", "rx_velocity",
    "distance (m)", "latency (ms)", "per_ue_loss_pct (%)", "ipg (ms)", "tx_family_id",
    "tx_equipment_id", "tx_seq_num", "tx_prioirty", "tx_timestamp (us)", "channel_busy_percentage",
    "tx_fix_mode", "tx_latitude", "tx_longitude", "tx_altitude (m)", "tx_quantity_SV_used",
    "tx_Semi_Major_Axis_Accuracy", "tx_heading", "tx_velocity", "time_confidence",
    "time_uncertainty", "position_confidence", "subframe#", "subchannel_index",
]
RTK_T0_MS = 1_770_298_260_000                      # CAM epoch (UTC): 2026-02-05 13:31:00
LOCAL_T0 = RTK_T0_MS / 1000 + LOCAL_TIME_OFFSET_S  # same instant in local time


def local_date(epoch_s):
    """Datalake $date string for a local epoch time (datalake stamps local time with a 'Z')."""
    return pd.Timestamp(epoch_s, unit="s").strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def write_cam(path, t0_ms=RTK_T0_MS, n=20, bad_line=False):
    """CAM log with one position per second; latitude 43.56 + i * 1e-4."""
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({
                "message_type": "cam",
                "source_uuid": "obu_test",
                "timestamp": t0_ms + i * 1000,
                "message": {"basic_container": {"reference_position_with_confidence": {
                    "latitude": 435_600_000 + i * 1000,
                    "longitude": 14_660_000 + i * 1000,
                }}},
            }) + "\n")
            if bad_line and i == 0:
                f.write("{not json\n")


def write_pc5_log(path, rx_us_list, latency=13.5):
    """PC5 receiver log with the SDK preamble and statistics lines around the data."""
    stats = "<1770297817142424> |    0 | +   0 packets |   0.00 packets per second (PPS)| 0.00 ms avg latency"
    lines = [
        "acme,SDK_ver#,SDK_build_date,build_time,direction",
        "\t,1,Oct  9 2022,02:58:25,Rx",
        "v2x_id:",
        ",".join(PC5_COLUMNS),
        " Epoch-ms     |  Tot-pkts   | New-pkts  |   PPS  | Latency | RV's | CBP %",
        stats,
    ]
    for seq, rx_us in enumerate(rx_us_list):
        values = {column: " " for column in PC5_COLUMNS}
        values.update({
            "rx_timestamp (us)": str(rx_us), "latency (ms)": f" {latency}",
            "per_ue_loss_pct (%)": "0.0", "ipg (ms)": "100.0", "tx_family_id": "81",
            "tx_equipment_id": "1", "tx_seq_num": str(seq), "tx_prioirty": "2",
            "tx_timestamp (us)": str(rx_us - 14000), "channel_busy_percentage": " 0% ",
            "subframe#": "", "subchannel_index": "",
        })
        lines.append(",".join(values[column] for column in PC5_COLUMNS))
        lines.append(stats)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def ue(ip, **cell_overrides):
    cell = {field: 1 for field in RADIO_CELL_FIELDS}
    cell.update(cell_overrides)
    return {"bearers": [{"ip": "192.168.4.6"}, {"ip": ip}], "cells": [cell]}


def iperf_interval(start):
    return {"sum": {"start": start, "end": start + 1, "bits_per_second": 300_000.0,
                    "jitter_ms": 2.5, "lost_packets": 0, "packets": 26}}


def rtk_frame(times_s, latitudes):
    return pd.DataFrame({
        "message_type": "cam",
        "source_uuid": "obu_test",
        "timestamp": pd.to_datetime(np.asarray(times_s) * 1000, unit="ms"),
        "latitude": latitudes,
        "longitude": [1.466] * len(latitudes),
    })


class TestSourceIngestion:
    """Per-source parsing, mirroring the cohda notebook cells."""

    def test_parse_pc5_log_reads_only_data_lines(self, tmp_path):
        path = tmp_path / "logs_pc5_01e0_050226_1323.log"
        rx = [1_770_298_265_000_000, 1_770_298_265_100_000]
        write_pc5_log(path, rx)

        df = parse_pc5_log(str(path))

        assert list(df.columns) == PC5_COLUMNS + ["timestamp"]
        assert len(df) == 2
        assert df["latency (ms)"].tolist() == [13.5, 13.5]
        # Text fields are stripped regardless of the pandas version
        assert df["rx_latitude"].tolist() == ["", ""]
        assert df["channel_busy_percentage"].tolist() == ["0%", "0%"]
        # UTC receive epoch -> local time
        assert df["timestamp"].tolist() == pytest.approx([r / 1e6 + LOCAL_TIME_OFFSET_S for r in rx])

    def test_ingest_pc5_uses_cohda_rsu_order_and_keeps_each_file_index(self, tmp_path):
        paths = []
        for rsu in ["zz99", "01e0", "0cb8"]:
            path = tmp_path / f"logs_pc5_{rsu}_050226_1323.log"
            write_pc5_log(path, [1_770_298_265_000_000, 1_770_298_265_100_000])
            paths.append(str(path))

        df = ingest_pc5(paths)

        assert list(df["Id"]) == ["0cb8", "0cb8", "01e0", "01e0", "zz99", "zz99"]
        assert list(df.index) == [0, 1, 0, 1, 0, 1]
        assert list(df.columns[-2:]) == ["timestamp", "Id"]

    def test_ingest_radio_skips_malformed_ues_and_keeps_datalake_time(self, tmp_path):
        entries = [
            {"timestamp": {"$date": "2026-02-05T14:28:00.769Z"},
             "ue_list": [ue("10.0.0.1", pusch_snr=12.5), {"bearers": [{"ip": "only-one"}], "cells": [{}]}]},
            {"timestamp": {"$date": "2026-02-05T14:28:01.769Z"}, "ue_list": [ue("10.0.0.2")]},
            {"message": "entry without timestamp"},
        ]
        path = tmp_path / "datalake._5G_UE.json"
        path.write_text(json.dumps(entries))

        df = ingest_radio(str(path))

        assert list(df.columns) == ["timestamp", "ip"] + RADIO_CELL_FIELDS
        assert df["ip"].tolist() == ["10.0.0.1", "10.0.0.2"]
        assert df["pusch_snr"].iloc[0] == 12.5
        assert epoch_seconds(df["timestamp"]).iloc[0] == pytest.approx(
            pd.Timestamp("2026-02-05T14:28:00.769Z").timestamp())

    def test_ingest_iperf_one_row_per_interval(self, tmp_path):
        entries = [
            {"rutx_ip": "10.0.0.1", "iperf": {"intervals": [iperf_interval(0), iperf_interval(1)]}},
            {"iperf": {"error": "control socket has closed unexpectedly"}},
        ]
        path = tmp_path / "datalake._5G_RUTX_iperf.json"
        path.write_text(json.dumps(entries))

        df = ingest_iperf(str(path))

        assert list(df.columns) == ["start_time", "end_time", "bitrate_mbps", "jitter_ms",
                                    "lost_packets", "packets", "ip"]
        assert len(df) == 2
        assert df["bitrate_mbps"].tolist() == pytest.approx([0.3, 0.3])
        assert df["ip"].tolist() == ["unknown", "unknown"]

    def test_ingest_ping_keeps_measured_pings_and_packet_loss(self, tmp_path):
        entries = [
            {"ip": "10.0.0.1", "ping": {"avg_latency": "49.230 ms", "packet_loss": "20%"},
             "timestamp": {"$date": "2026-02-05T14:31:05.000Z"}},
            {"ip": "10.0.0.1", "ping": {"packet_loss": "100%"},
             "timestamp": {"$date": "2026-02-05T14:31:07.000Z"}},
            {"ip": "10.0.0.1", "ping": {"avg_latency": "36.5 ms"},
             "timestamp": {"$date": "2026-02-05T14:31:09.000Z"}},
        ]
        path = tmp_path / "datalake._5G_RUTX_ping.json"
        path.write_text(json.dumps(entries))

        df = ingest_ping(str(path))

        assert list(df.columns) == ["timestamp", "test_type", "latency_ms", "ip", "packet_loss"]
        assert df["latency_ms"].tolist() == [49.23, 36.5]
        assert df["packet_loss"].iloc[0] == 20.0
        assert np.isnan(df["packet_loss"].iloc[1])

    def test_ingest_rtk_skips_bad_lines_and_uses_local_time(self, tmp_path):
        path = tmp_path / "CAM_obu_20260205_142622.jsonl"
        write_cam(path, n=3, bad_line=True)

        df = ingest_rtk(str(path))

        assert list(df.columns) == ["message_type", "source_uuid", "timestamp", "latitude", "longitude"]
        assert len(df) == 3
        assert epoch_seconds(df["timestamp"]).tolist() == pytest.approx([LOCAL_T0 + i for i in range(3)])
        assert df["latitude"].tolist() == pytest.approx([43.56 + i * 1e-4 for i in range(3)])

    def test_epoch_seconds_ignores_unit_and_timezone(self):
        instant = pd.Timestamp("2026-02-05T13:31:00.250Z")
        variants = [
            pd.Series([instant]),
            pd.Series([instant.tz_localize(None)]).astype("datetime64[ms]"),
            pd.Series([instant.tz_localize(None)]).astype("datetime64[ns]"),
        ]
        for series in variants:
            assert epoch_seconds(series).iloc[0] == pytest.approx(instant.timestamp())


class TestAttachRtkGps:
    """Nearest-RTK position matching."""

    def test_rows_keep_their_time_and_match_within_tolerance(self):
        rtk = rtk_frame([LOCAL_T0 + i * 10 for i in range(5)], [43.56 + i * 1e-4 for i in range(5)])
        ts = np.array([
            LOCAL_T0 - 500,                               # before RTK coverage: no position
            LOCAL_T0 - GPS_MATCH_TOLERANCE_S,             # exactly at the tolerance: matched
            LOCAL_T0 + 11,                                # nearest sample is LOCAL_T0 + 10
            LOCAL_T0 + 40 + GPS_MATCH_TOLERANCE_S + 0.5,  # just beyond the tolerance
        ])

        out = attach_rtk_gps(pd.DataFrame({"Id": ["0cb8"] * 4}), ts, rtk)

        np.testing.assert_allclose(out["timestamp"], ts)
        latitude = out["Latitude"].to_numpy()
        assert np.isnan(latitude[0])
        assert latitude[1] == pytest.approx(43.56)
        assert latitude[2] == pytest.approx(43.5601)
        assert np.isnan(latitude[3])

    def test_equidistant_rows_take_the_earliest_rtk_row(self):
        # RTK rows out of time order, with a duplicated time: idxmin picks the first row
        rtk = rtk_frame([LOCAL_T0 + 2, LOCAL_T0, LOCAL_T0 + 2], [43.50, 43.51, 43.52])

        out = attach_rtk_gps(pd.DataFrame({"x": [0, 1]}), np.array([LOCAL_T0 + 1, LOCAL_T0 + 3]), rtk)

        assert out["Latitude"].tolist() == pytest.approx([43.50, 43.50])

    def test_rows_sharing_index_labels_keep_their_own_positions(self):
        rtk = rtk_frame([LOCAL_T0 + i for i in range(10)], [43.56 + i * 1e-4 for i in range(10)])
        pc5 = pd.concat([pd.DataFrame({"Id": ["0cb8", "0cb8"]}), pd.DataFrame({"Id": ["01e0", "01e0"]})])

        out = attach_rtk_gps(pc5, np.array([LOCAL_T0 + 1, LOCAL_T0 + 2, LOCAL_T0 + 7, LOCAL_T0 + 8]), rtk)

        assert out["Latitude"].tolist() == pytest.approx([43.5601, 43.5602, 43.5607, 43.5608])
        assert list(out.index) == [0, 1, 0, 1]

    def test_without_rtk_rows_get_no_position(self):
        out = attach_rtk_gps(pd.DataFrame({"x": [0]}), np.array([LOCAL_T0]), None)

        assert out["timestamp"].iloc[0] == LOCAL_T0
        assert np.isnan(out["Latitude"].iloc[0])
        assert np.isnan(out["Longitude"].iloc[0])


class TestIngestDataset:
    """Folder-level ingestion, tour layout and merge."""

    @staticmethod
    def write_tour(tour_dir, offset_s=0):
        tour_dir.mkdir(parents=True)
        t0_ms = RTK_T0_MS + offset_s * 1000
        utc0 = t0_ms / 1000
        local0 = utc0 + LOCAL_TIME_OFFSET_S
        write_cam(tour_dir / "CAM_obu_test.jsonl", t0_ms=t0_ms)

        pings = [{"ip": "10.0.0.1", "ping": {"avg_latency": "40.0 ms", "packet_loss": "0%"},
                  "timestamp": {"$date": local_date(local0 + 2 * i)}} for i in range(3)]
        (tour_dir / "datalake._5G_RUTX_ping.json").write_text(json.dumps(pings))
        (tour_dir / "datalake._5G_UE.json").write_text(json.dumps([
            {"timestamp": {"$date": local_date(local0)}, "ue_list": [ue("10.0.0.1")]},
        ]))
        (tour_dir / "datalake._5G_RUTX_iperf.json").write_text(json.dumps([
            {"iperf": {"intervals": [iperf_interval(0)]}},
        ]))
        write_pc5_log(tour_dir / "logs_pc5_0cb8_x.log", [int((utc0 + 1 + i) * 1e6) for i in range(4)])
        write_pc5_log(tour_dir / "logs_pc5_01e0_x.log", [int((utc0 + 3 + i) * 1e6) for i in range(2)])
        return local0

    def test_ingest_dataset_writes_each_tour_then_merges(self, tmp_path):
        raw = tmp_path / "raw"
        local0 = self.write_tour(raw / "Tour1")
        self.write_tour(raw / "Tour2", offset_s=86_400)
        (raw / "notes").mkdir()  # folders without raw logs are not tours
        out = tmp_path / "out"

        written = ingest_dataset(str(raw), str(out))

        assert written == [str(out / "Tour1"), str(out / "Tour2"), str(out)]
        expected = ["df_iperf.json", "df_pc5.json", "df_ping.json", "df_radio.json", "df_rtk.json"]
        for folder in (out / "Tour1", out / "Tour2", out):
            assert sorted(f for f in os.listdir(folder) if f.startswith("df_")) == expected

        # Tour files store ping/PC5 timestamps as epoch seconds, like the notebook
        tour_pc5 = json.loads((out / "Tour1" / "df_pc5.json").read_text())
        ts_column = tour_pc5["columns"].index("timestamp")
        assert isinstance(tour_pc5["data"][0][ts_column], float)
        assert tour_pc5["index"] == [0, 1, 2, 3, 0, 1]

        # Ping times are the datalake $date, unshifted
        tour_ping = json.loads((out / "Tour1" / "df_ping.json").read_text())
        ping_ts = [row[tour_ping["columns"].index("timestamp")] for row in tour_ping["data"]]
        assert ping_ts == pytest.approx([local0 + 2 * i for i in range(3)])

        # The merge re-reads and concatenates, storing epoch milliseconds
        merged_pc5 = json.loads((out / "df_pc5.json").read_text())
        assert isinstance(merged_pc5["data"][0][merged_pc5["columns"].index("timestamp")], int)
        assert merged_pc5["index"] == list(range(12))

        ping = pd.read_json(str(out / "Tour1" / "df_ping.json"), orient="split")
        assert list(ping.columns) == ["timestamp", "test_type", "latency_ms", "ip",
                                      "Latitude", "Longitude", "packet_loss"]
        assert ping["Latitude"].tolist() == pytest.approx([43.56, 43.5602, 43.5604])

        pc5 = pd.read_json(str(out / "df_pc5.json"), orient="split")
        assert len(pc5) == 12
        assert pc5["Latitude"].notna().all()

    def test_ingest_dataset_without_raw_logs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_dataset(str(tmp_path), str(tmp_path / "out"))
