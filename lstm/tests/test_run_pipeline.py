"""
Tests for run_pipeline.py - CLI argument handling.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_RATS, DEFAULT_RATS
from run_pipeline import parse_args


class TestRatsArgument:
    """Tests for the --rats option shared by all pipeline stages."""

    def test_default_rats_exclude_dsrc(self):
        args = parse_args(["--data", "somewhere"])
        assert args.rats == DEFAULT_RATS
        assert "dsrc" not in args.rats

    def test_rats_normalized(self):
        args = parse_args(["--data", "somewhere", "--rats", "5g", "dsrc", "pc5", "5g"])
        assert args.rats == ALL_RATS

    def test_unknown_rat_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--data", "somewhere", "--rats", "5g", "wifi"])


class TestJsonTours:
    """Tours of df_*.json exports (no raw logs) as a data source."""

    def test_raw_data_and_json_tours_are_exclusive(self):
        with pytest.raises(SystemExit):
            parse_args(["--raw_data", "raw", "--json_tours", "tours"])

    def test_stage_trim_converts_tours_into_default_folder(self, tmp_path):
        from unittest.mock import patch
        from run_pipeline import stage_trim

        args = parse_args(["--json_tours", str(tmp_path)])
        with patch("scripts.convert_json_to_trim.convert_tours") as convert:
            stage_trim(args)

        expected = os.path.join(str(tmp_path), "trimmed")
        convert.assert_called_once_with(str(tmp_path), expected)
        assert args.data == expected


class TestStageMatchFreshness:
    """Matching re-runs when matched CSVs predate the trimmed CSVs."""

    @staticmethod
    def write_files(folder, trimmed_mtime, matched_mtime):
        for name in ("trim_5g.csv", "trim_pc5.csv"):
            path = folder / name
            path.write_text("x\n")
            os.utime(path, (trimmed_mtime, trimmed_mtime))
        for name in ("matched_5g.csv", "matched_pc5.csv", "super.csv"):
            path = folder / name
            path.write_text("x\n")
            os.utime(path, (matched_mtime, matched_mtime))

    def run_stage(self, folder):
        from unittest.mock import patch
        from run_pipeline import stage_match

        with patch("scripts.prepare_data.match_data") as match:
            stage_match(parse_args(["--data", str(folder)]))
        return match

    def test_stale_matched_files_are_rebuilt(self, tmp_path):
        self.write_files(tmp_path, trimmed_mtime=2_000_000, matched_mtime=1_000_000)
        match = self.run_stage(tmp_path)
        match.assert_called_once_with(str(tmp_path), rats=DEFAULT_RATS)

    def test_up_to_date_matched_files_are_kept(self, tmp_path):
        self.write_files(tmp_path, trimmed_mtime=1_000_000, matched_mtime=2_000_000)
        self.run_stage(tmp_path).assert_not_called()
