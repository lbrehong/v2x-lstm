"""
Tests for scripts/convert_json_to_trim.py - df_*.json to trim_*.csv conversion.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.convert_json_to_trim import _ts_to_epoch_ms, convert_5g

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

    def test_joins_radio_by_ip_and_uses_packet_loss_pdr(self, tmp_path):
        write_split_json(ping_frame(), tmp_path / "df_ping.json")
        write_split_json(pd.DataFrame({
            "timestamp": [T0 - pd.Timedelta(seconds=0.5),   # 0.5 s before ping 0
                          T0 + pd.Timedelta(seconds=4.2),   # 0.2 s after ping 2
                          T0 + pd.Timedelta(seconds=2)],    # other UE, same time as ping 1
            "ip": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
            "pusch_snr": [10.0, 20.0, 99.0],
            "ul_path_loss": [100.0, 110.0, 50.0],
        }), tmp_path / "df_radio.json")

        out = convert_5g(str(tmp_path), None)

        assert list(out.columns) == ["tx_seq_num", "tx_timestamp_ms", "tx_latitude", "tx_longitude",
                                     "latency_ms", "sinr", "rsrp", "pdr"]
        assert out["latency_ms"].tolist() == [40.0, 41.0, 42.0, 43.0]
        assert out["pdr"].tolist() == pytest.approx([1.0, 0.8, 0.6, 1.0])
        # Ping 1 has no radio sample of its own UE within 2 s: filled from ping 0
        assert out["sinr"].iloc[:3].tolist() == [10.0, 10.0, 20.0]
        assert out["rsrp"].iloc[:3].tolist() == [-100.0, -100.0, -110.0]
        # A UE without any radio sample stays empty
        assert np.isnan(out["sinr"].iloc[3])
        assert np.isnan(out["rsrp"].iloc[3])

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


def test_ts_to_epoch_ms_ignores_unit_and_timezone():
    ts = pd.Series(pd.to_datetime(["2026-02-05 13:31:00.250"]))
    expected = pd.Timestamp("2026-02-05 13:31:00.250", tz="UTC").timestamp() * 1000

    for series in (ts.astype("datetime64[ns]"), ts.astype("datetime64[ms]"), ts.dt.tz_localize("UTC")):
        assert _ts_to_epoch_ms(series).iloc[0] == pytest.approx(expected)
