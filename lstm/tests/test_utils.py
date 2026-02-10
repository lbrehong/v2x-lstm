"""
Tests for utils.py - Utility functions for file discovery and model loading.
"""
import pytest
import os
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import find_files_with_string, get_latest_model, ensure_dir_exists


class TestFindFilesWithString:
    """Tests for the find_files_with_string function."""

    def test_find_matching_files(self, tmp_path):
        """Should find files containing the search string."""
        # Create test files
        (tmp_path / "lstm_5g_12345.keras").touch()
        (tmp_path / "lstm_pc5_12346.keras").touch()
        (tmp_path / "gru_5g_12347.keras").touch()
        (tmp_path / "other_file.txt").touch()

        result = find_files_with_string(str(tmp_path), "lstm")

        assert len(result) == 2
        assert "lstm_5g_12345.keras" in result
        assert "lstm_pc5_12346.keras" in result
        assert "gru_5g_12347.keras" not in result

    def test_find_no_matching_files(self, tmp_path):
        """Should return empty list when no files match."""
        (tmp_path / "gru_5g_12345.keras").touch()
        (tmp_path / "rnn_pc5_12346.keras").touch()

        result = find_files_with_string(str(tmp_path), "lstm")

        assert result == []

    def test_find_files_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        result = find_files_with_string(str(tmp_path), "lstm")

        assert result == []

    def test_find_files_partial_match(self, tmp_path):
        """Should find files with partial string match."""
        (tmp_path / "model_5g_v1.keras").touch()
        (tmp_path / "model_5g_v2.keras").touch()
        (tmp_path / "model_pc5_v1.keras").touch()

        result = find_files_with_string(str(tmp_path), "5g")

        assert len(result) == 2
        assert "model_5g_v1.keras" in result
        assert "model_5g_v2.keras" in result


class TestGetLatestModel:
    """Tests for the get_latest_model function."""

    def test_get_latest_model_single_file(self, tmp_path):
        """Should return the only matching model file."""
        (tmp_path / "lstm_5g_1000000.keras").touch()

        result = get_latest_model("lstm", "5g", str(tmp_path))

        assert result is not None
        assert "lstm_5g_1000000.keras" in result

    def test_get_latest_model_multiple_files(self, tmp_path):
        """Should return the model with highest timestamp."""
        (tmp_path / "lstm_5g_1000000.keras").touch()
        (tmp_path / "lstm_5g_1500000.keras").touch()
        (tmp_path / "lstm_5g_1200000.keras").touch()

        result = get_latest_model("lstm", "5g", str(tmp_path))

        assert result is not None
        assert "lstm_5g_1500000.keras" in result

    def test_get_latest_model_no_matching_files(self, tmp_path):
        """Should return None when no files match the pattern."""
        (tmp_path / "gru_5g_1000000.keras").touch()
        (tmp_path / "lstm_pc5_1000000.keras").touch()

        result = get_latest_model("lstm", "5g", str(tmp_path))

        assert result is None

    def test_get_latest_model_empty_directory(self, tmp_path):
        """Should return None for empty directory."""
        result = get_latest_model("lstm", "5g", str(tmp_path))

        assert result is None

    def test_get_latest_model_different_rats(self, tmp_path):
        """Should distinguish between different RAT types."""
        (tmp_path / "lstm_5g_1000000.keras").touch()
        (tmp_path / "lstm_pc5_2000000.keras").touch()  # Higher timestamp but different RAT
        (tmp_path / "lstm_dsrc_1500000.keras").touch()

        result_5g = get_latest_model("lstm", "5g", str(tmp_path))
        result_pc5 = get_latest_model("lstm", "pc5", str(tmp_path))
        result_dsrc = get_latest_model("lstm", "dsrc", str(tmp_path))

        assert "lstm_5g_1000000.keras" in result_5g
        assert "lstm_pc5_2000000.keras" in result_pc5
        assert "lstm_dsrc_1500000.keras" in result_dsrc

    def test_get_latest_model_different_model_types(self, tmp_path):
        """Should distinguish between different model types."""
        (tmp_path / "lstm_5g_1000000.keras").touch()
        (tmp_path / "gru_5g_2000000.keras").touch()
        (tmp_path / "rnn_5g_1500000.keras").touch()

        result_lstm = get_latest_model("lstm", "5g", str(tmp_path))
        result_gru = get_latest_model("gru", "5g", str(tmp_path))
        result_rnn = get_latest_model("rnn", "5g", str(tmp_path))

        assert "lstm_5g_1000000.keras" in result_lstm
        assert "gru_5g_2000000.keras" in result_gru
        assert "rnn_5g_1500000.keras" in result_rnn

    def test_get_latest_model_ignores_non_matching_format(self, tmp_path):
        """Should ignore files that don't match the expected naming pattern."""
        (tmp_path / "lstm_5g_1000000.keras").touch()
        (tmp_path / "lstm_5g_invalid.keras").touch()  # No timestamp
        (tmp_path / "lstm_5g.keras").touch()  # No timestamp

        result = get_latest_model("lstm", "5g", str(tmp_path))

        assert result is not None
        assert "lstm_5g_1000000.keras" in result


class TestEnsureDirExists:
    """Tests for the ensure_dir_exists function."""

    def test_create_new_directory(self, tmp_path):
        """Should create directory that doesn't exist."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()

        ensure_dir_exists(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory_unchanged(self, tmp_path):
        """Should not raise error for existing directory."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        (existing_dir / "test_file.txt").write_text("content")

        ensure_dir_exists(str(existing_dir))

        assert existing_dir.exists()
        assert (existing_dir / "test_file.txt").exists()

    def test_create_nested_directories(self, tmp_path):
        """Should create nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        ensure_dir_exists(str(nested_dir))

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_empty_string_path(self, tmp_path):
        """Should handle empty string path - may raise error on some systems."""
        # Empty string path behavior varies by OS
        # On Linux, os.makedirs("") raises FileNotFoundError
        # This test verifies that ensure_dir_exists doesn't handle empty string
        with pytest.raises(FileNotFoundError):
            ensure_dir_exists("")
