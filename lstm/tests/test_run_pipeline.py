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
