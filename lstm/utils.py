"""
Shared utility functions for the RAT prediction project.
"""
import os
import glob
import re
from typing import Optional, List

from config import MODEL_DIR


def find_files_with_string(directory: str, search_string: str) -> List[str]:
    """
    Find files in a directory matching a substring pattern.

    Args:
        directory: Path to search in
        search_string: Substring to match in filenames

    Returns:
        List of matching filenames (not full paths)
    """
    all_files = os.listdir(directory)
    matching_files = [f for f in all_files if search_string in f]
    return matching_files


def get_latest_model(model_type: str, rat: str, model_dir: str = MODEL_DIR) -> Optional[str]:
    """
    Find the most recent model file based on the model type and RAT.

    Args:
        model_type: Type of model ('lstm', 'gru', 'rnn')
        rat: RAT type ('5g', 'pc5', 'dsrc')
        model_dir: Directory containing saved models

    Returns:
        Path to most recent model, or None if not found
    """
    pattern = os.path.join(model_dir, f"{model_type}_{rat}_*.keras")
    model_files = glob.glob(pattern)

    regex = re.compile(rf"{model_type}_{rat}_(\d+)\.keras")
    files_with_time = []

    for filepath in model_files:
        match = regex.search(os.path.basename(filepath))
        if match:
            files_with_time.append((filepath, int(match.group(1))))

    if not files_with_time:
        print("No existing models found.")
        return None

    latest_model = max(files_with_time, key=lambda x: x[1])[0]
    print(f"Found a model: {latest_model}")
    return latest_model


def ensure_dir_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
