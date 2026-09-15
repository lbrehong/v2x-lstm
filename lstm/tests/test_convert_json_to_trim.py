"""
Tests for scripts/convert_json_to_trim.py - df_*.json to trim_*.csv conversion.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.convert_json_to_trim import (
    _load_df_json, _ts_to_epoch_ms, convert_5g, convert_pc5, convert_tours, find_tour_dirs,
)

T0 = pd.Timestamp("2026-02-05 14:31:00")  # datalake time, shared by df_ping and df_radio


def write_split_json(df, path):
    df.to_json(path, orient="split", compression="infer", date_format="epoch")


def ping_frame():
    """Four pings every 2 s; the last one from a UE without radio samples."""
    return pd.DataFrame({
        "timestamp": [T0 + pd.Timedelta(seconds=2 * i) for i in range(4)],
        "test_type": "ping",
        "latency_ms": [40.0, 41.0, 42.0, 43.0],
        "ip": ["10.0.0.1", "10.0.0.1", "10.0.0.1", "10.0.0.9"],
        "Latitude": 43.56,
        "Longitude": 1.466,
        "packet_loss": [0.0, 20.0, 40.0, 0.0],
    })


class TestConvert5g:

    @staticmethod
    def write_radio(tmp_path):
        write_split_json(pd.DataFrame({
            "timestamp": [T0 - pd.Timedelta(seconds=0.5),   # 0.5 s before ping 0
                          T0 + pd.Timedelta(seconds=4.2),   # 0.2 s after ping 2
                          T0 + pd.Timedelta(seconds=2)],    # other UE, same time as ping 1
            "ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "pusch_snr": [10.0, 20.0, 99.0],
            "ul_path_loss": [100.0, 110.0, 50.0],
        }), tmp_path / "df_radio.json")

    def test_packet_loss_pdr_only_when_enabled(self, tmp_path, monkeypatch):
        write_split_json(ping_frame(), tmp_path / "df_ping.json")

        # Default: no pdr column, a rolling count is computed downstream for every dataset
        assert "pdr" not in convert_5g(str(tmp_path), None).columns

        monkeypatch.setattr("scripts.convert_json_to_trim.FIVEG_PDR_FROM_PACKET_LOSS", True)
        out = convert_5g(str(tmp_path), None)
        assert out["pdr"].tolist() == pytest.approx([1.0, 0.8, 0.6, 1.0])

    def test_joins_radio_by_ip(self, tmp_path):
        write_split_json(ping_frame(), tmp_path / "df_ping.json")
        self.write_radio(tmp_path)

        out = convert_5g(str(tmp_path), None)

        assert list(out.columns) == ["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
                                     "latency_ms", "sinr", "rsrp"]
        assert out["latency_ms"].tolist() == [40.0, 41.0, 42.0, 43.0]
        # Each ping takes its UE's nearest radio sample: ping 1 is 2.5 s after the
        # first sample and 2.2 s before the second; the other UE's sample is ignored
        assert out["sinr"].iloc[:3].tolist() == [10.0, 20.0, 20.0]
        assert out["rsrp"].iloc[:3].tolist() == [-100.0, -110.0, -110.0]
        # A UE without any radio sample stays empty
        assert np.isnan(out["sinr"].iloc[3])
        assert np.isnan(out["rsrp"].iloc[3])

    def test_pings_far_from_any_radio_sample_get_no_radio_metrics(self, tmp_path):
        """A radio log that stops early must not lend stale values to later pings."""
        from scripts.convert_json_to_trim import RADIO_MATCH_TOLERANCE_MS
        ping = ping_frame()
        late = pd.Timedelta(milliseconds=RADIO_MATCH_TOLERANCE_MS) + pd.Timedelta(minutes=5)
        ping["timestamp"] = [T0, T0 + pd.Timedelta(seconds=2), T0 + late, T0 + late + pd.Timedelta(seconds=2)]
        ping["ip"] = "10.0.0.1"
        write_split_json(ping, tmp_path / "df_ping.json")
        write_split_json(pd.DataFrame({
            "timestamp": [T0 - pd.Timedelta(seconds=0.5)],
            "ip": ["10.0.0.1"], "pusch_snr": [10.0], "ul_path_loss": [100.0],
        }), tmp_path / "df_radio.json")

        out = convert_5g(str(tmp_path), None)

        assert out["sinr"].tolist()[:2] == [10.0, 10.0]
        assert out["sinr"].iloc[2:].isna().all() and out["rsrp"].iloc[2:].isna().all()

    def test_without_packet_loss_or_radio_has_no_pdr_column(self, tmp_path):
        write_split_json(ping_frame().drop(columns=["packet_loss"]), tmp_path / "df_ping.json")

        out = convert_5g(str(tmp_path), None)

        assert "pdr" not in out.columns
        assert out["sinr"].isna().all()
        assert len(out) == 4

    def test_clips_latency_to_the_5g_bound(self, tmp_path):
        from config import LATENCY_BOUNDS
        cap = LATENCY_BOUNDS["5g"][1]
        ping = ping_frame()
        ping["latency_ms"] = [40.0, cap - 1, cap + 1, 6000.0]
        write_split_json(ping, tmp_path / "df_ping.json")

        out = convert_5g(str(tmp_path), None)

        # Stalls are kept, at the bound
        assert out["latency_ms"].tolist() == [40.0, cap - 1, cap, cap]


def test_load_df_json_tolerates_out_of_range_integers(tmp_path):
    """cohda's df_radio.json holds 2^64 sentinels that pandas' JSON reader rejects."""
    good = pd.DataFrame({
        "timestamp": [T0, T0 + pd.Timedelta(seconds=1)],
        "ip": ["10.0.0.1", "10.0.0.1"],
        "dl_bitrate": [100, 200],
        "pusch_snr": [10.5, 11.5],
    })
    write_split_json(good, tmp_path / "good.json")
    text = (tmp_path / "good.json").read_text()
    raw = json.loads(text)
    raw["data"][1][2] = 18446744073709551616  # 2^64
    (tmp_path / "bad.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError):
        pd.read_json(tmp_path / "bad.json", orient="split")
    loaded = _load_df_json(str(tmp_path / "bad.json"))
    reference = _load_df_json(str(tmp_path / "good.json"))

    assert loaded["dl_bitrate"].iloc[0] == 100 and np.isnan(loaded["dl_bitrate"].iloc[1])
    assert loaded["timestamp"].dtype == reference["timestamp"].dtype
    assert loaded["timestamp"].tolist() == reference["timestamp"].tolist()
    assert loaded["pusch_snr"].tolist() == [10.5, 11.5]


def pc5_frame(t0_us, lat=43.56):
    return pd.DataFrame({
        "tx_seq_num": [1, 2, 3],
        "tx_timestamp (us)": [t0_us, t0_us + 100_000, t0_us + 200_000],
        "tx_latitude": [lat, np.nan, lat],
        "tx_longitude": [1.466, np.nan, 1.466],
        "latency (ms)": [10.0, 11.0, 12.0],
    })


def test_convert_pc5_uses_local_timestamp_when_present(tmp_path):
    """ingest_logs' local `timestamp` puts PC5 on the ping clock; raw tx epochs are 1 h off."""
    frame = pc5_frame(int((T0 - pd.Timedelta(hours=1)).timestamp() * 1e6))
    frame["timestamp"] = [T0, T0 + pd.Timedelta(milliseconds=100), T0 + pd.Timedelta(milliseconds=200)]
    write_split_json(frame, tmp_path / "df_pc5.json")
    expected = pd.Timestamp(T0, tz="UTC").timestamp() * 1000

    out = convert_pc5(str(tmp_path))

    assert out["tx_timestamp_ms"].iloc[0] == pytest.approx(expected)


def test_convert_pc5_without_local_timestamp_uses_tx_epoch(tmp_path):
    t0_us = 1_750_000_000_000_000
    write_split_json(pc5_frame(t0_us), tmp_path / "df_pc5.json")

    out = convert_pc5(str(tmp_path))

    assert out["tx_timestamp_ms"].iloc[0] == pytest.approx(t0_us / 1000)


def test_convert_pc5_drops_rows_without_transmitter_position(tmp_path):
    write_split_json(pc5_frame(1_750_000_000_000_000), tmp_path / "df_pc5.json")

    out = convert_pc5(str(tmp_path))

    assert out["latency_ms"].tolist() == [10.0, 12.0]
    assert out["tx_latitude"].notna().all()


class TestConvertTours:

    @staticmethod
    def write_tour(path, t0):
        path.mkdir(parents=True)
        ping = ping_frame()
        ping["timestamp"] = [t0 + pd.Timedelta(seconds=2 * i) for i in range(4)]
        write_split_json(ping, path / "df_ping.json")
        write_split_json(pc5_frame(int(t0.timestamp() * 1e6)), path / "df_pc5.json")

    def test_converts_each_tour_then_merges_in_time_order(self, tmp_path):
        root = tmp_path / "matched"
        later, earlier = T0 + pd.Timedelta(hours=3), T0
        self.write_tour(root / "combined" / "Tour1", later)
        self.write_tour(root / "combined" / "Tour2", earlier)
        out = root / "combined" / "trimmed"
        out.mkdir(parents=True)
        write_split_json(ping_frame(), out / "df_ping.json")  # decoy: output folder is not a tour

        tours = convert_tours(str(root), str(out))

        assert tours == [str(root / "combined" / "Tour1"), str(root / "combined" / "Tour2")]
        assert (out / "combined_Tour1" / "trim_5g.csv").exists()
        assert (out / "combined_Tour2" / "trim_pc5.csv").exists()
        merged_5g = pd.read_csv(out / "trim_5g.csv")
        merged_pc5 = pd.read_csv(out / "trim_pc5.csv")
        assert len(merged_5g) == 8 and len(merged_pc5) == 4
        assert merged_5g["tx_timestamp_ms"].is_monotonic_increasing
        assert merged_pc5["tx_timestamp_ms"].is_monotonic_increasing
        # Tour2 happened first, so its rows come first
        tour2_5g = pd.read_csv(out / "combined_Tour2" / "trim_5g.csv")
        assert merged_5g["tx_timestamp_ms"].iloc[:4].tolist() == tour2_5g["tx_timestamp_ms"].tolist()

    def test_finds_tours_directly_under_root(self, tmp_path):
        self.write_tour(tmp_path / "Tour1", T0)
        (tmp_path / "logs").mkdir()

        assert find_tour_dirs(str(tmp_path)) == [str(tmp_path / "Tour1")]

    def test_no_tours_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            convert_tours(str(tmp_path), str(tmp_path / "out"))


def test_ts_to_epoch_ms_ignores_unit_and_timezone():
    ts = pd.Series(pd.to_datetime(["2026-02-05 13:31:00.250"]))
    expected = pd.Timestamp("2026-02-05 13:31:00.250", tz="UTC").timestamp() * 1000

    for series in (ts.astype("datetime64[ns]"), ts.astype("datetime64[ms]"), ts.dt.tz_localize("UTC")):
        assert _ts_to_epoch_ms(series).iloc[0] == pytest.approx(expected)
