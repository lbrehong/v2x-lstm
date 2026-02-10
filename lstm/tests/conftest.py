"""
Pytest configuration and shared fixtures for RAT prediction tests.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_gps_data():
    """Sample GPS coordinates within Toulouse test area bounds."""
    return pd.DataFrame({
        'tx_latitude': [43.560, 43.562, 43.564, 43.566, 43.568],
        'tx_longitude': [1.465, 1.466, 1.467, 1.468, 1.469]
    })


@pytest.fixture
def sample_5g_data():
    """Sample 5G RAT data with all required features."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'tx_latitude': np.linspace(43.555, 43.567, n_samples),
        'tx_longitude': np.linspace(1.464, 1.471, n_samples),
        'latency_ms': np.random.uniform(5, 40, n_samples),
        'sinr': np.random.uniform(210, 360, n_samples),
        'rsrp': np.random.uniform(-120, -70, n_samples),
        'pdr': np.random.uniform(0.8, 1.0, n_samples),
        'tx_timestamp_ms': np.arange(0, n_samples * 20, 20),  # 20ms intervals
    })


@pytest.fixture
def sample_pc5_data():
    """Sample PC5 RAT data with all required features."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'tx_latitude': np.linspace(43.555, 43.567, n_samples),
        'tx_longitude': np.linspace(1.464, 1.471, n_samples),
        'latency_ms': np.random.uniform(5, 35, n_samples),
        'pdr': np.random.uniform(0.85, 1.0, n_samples),
        'tx_timestamp_ms': np.arange(0, n_samples * 20, 20),
    })


@pytest.fixture
def sample_dsrc_data():
    """Sample DSRC RAT data with all required features."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'tx_latitude': np.linspace(43.555, 43.567, n_samples),
        'tx_longitude': np.linspace(1.464, 1.471, n_samples),
        'rsrp_1': np.random.uniform(-140, -50, n_samples),
        'rsrp_2': np.random.uniform(-140, -50, n_samples),
        'latency_ms': np.random.uniform(5, 30, n_samples),
        'pdr': np.random.uniform(0.9, 1.0, n_samples),
        'tx_timestamp_ms': np.arange(0, n_samples * 20, 20),
    })


@pytest.fixture
def sample_rat_selection_row():
    """Sample DataFrame row for RAT selection testing."""
    return pd.Series({
        'pred_latency_ms_dsrc_lstm': 15.0,
        'pred_pdr_dsrc_lstm': 0.995,
        'pdr_dsrc': 0.99,
        'latency_ms_dsrc': 14.0,
        'pred_latency_ms_pc5_lstm': 12.0,
        'pred_pdr_pc5_lstm': 0.992,
        'pdr_pc5': 0.985,
        'latency_ms_pc5': 11.0,
        'pred_latency_ms_5g_lstm': 18.0,
        'pred_pdr_5g_lstm': 0.998,
        'pdr_5g': 0.99,
        'latency_ms_5g': 17.0,
    })


@pytest.fixture
def sample_sequences():
    """Sample LSTM sequences for model testing."""
    np.random.seed(42)
    n_samples = 50
    timesteps = 10
    features = 6
    X = np.random.rand(n_samples, timesteps, features)
    y = np.random.rand(n_samples, 2)  # latency, pdr
    return X, y
