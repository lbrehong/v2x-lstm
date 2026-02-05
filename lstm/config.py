"""
Centralized configuration for the RAT prediction project.

All constants, bounds, and hyperparameters are defined here
to ensure consistency across modules.
"""
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# GPS Bounds (Toulouse, France - test area)
# =============================================================================
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176

# =============================================================================
# Signal Quality Bounds
# =============================================================================
MIN_LATENCY, MAX_LATENCY = 4, 50  # in ms
MIN_PDR, MAX_PDR = 0.0, 1.0
MIN_THROUGHPUT, MAX_THROUGHPUT = 0, 100  # in Mbps
MIN_SINR_5G, MAX_SINR_5G = 200, 375
MIN_RSRP_5G, MAX_RSRP_5G = -127, -67
MIN_RSRP_DSRC, MAX_RSRP_DSRC = -150, -45

# =============================================================================
# Training Hyperparameters
# =============================================================================
TIMESTEPS = 10
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 5
SAMPLES = 20000  # number of samples for testing
TRAIN_RATIO = 0.4

# =============================================================================
# Data Collection Parameters
# =============================================================================
TX_INTERVAL_MS = 20  # in ms, 50 packets per second
PDR_WINDOW = 10  # in seconds

# =============================================================================
# RAT Selection Thresholds
# =============================================================================
PDR_RELIABILITY_THRESHOLD = 0.99  # Minimum PDR for reliable transmission
PDR_AVAILABILITY_THRESHOLD = 0.1  # Minimum PDR to consider RAT available
LATENCY_TIE_MARGIN_MS = 1.0  # Latency difference to trigger tie-breaking

# =============================================================================
# Packet Size Bounds (for queue simulator integration)
# =============================================================================
PACKET_SIZE_BOUNDS = {
    "dsrc": {"min": 100, "max": 1500, "levels": 8},
    "pc5": {"min": 100, "max": 2000, "levels": 8},
    "5g": {"min": 100, "max": 1400, "levels": 8},
}

# DTMC (Discrete Time Markov Chain) Packet Sizing Parameters
DTMC_THRESHOLD_HIGH = 0.95  # PDR above this may increase packet size
DTMC_THRESHOLD_LOW = 0.85   # PDR below this may decrease packet size
DTMC_P_UP = 0.7             # Probability of increasing when PDR high
DTMC_P_DOWN = 0.8           # Probability of decreasing when PDR low

# =============================================================================
# Feature Columns by RAT Type
# =============================================================================
FEATURE_COLS = {
    '5g': ['tx_latitude', 'tx_longitude', 'latency_ms', 'sinr', 'rsrp', 'pdr'],
    'pc5': ['tx_latitude', 'tx_longitude', 'latency_ms', 'pdr'],
    'dsrc': ['tx_latitude', 'tx_longitude', 'rsrp_1', 'rsrp_2', 'latency_ms', 'pdr'],
}

FEATURES_COUNT = {
    '5g': 6,
    'pc5': 4,
    'dsrc': 6,
}

TARGET_COLS = ['latency_ms', 'pdr']

# =============================================================================
# Directories
# =============================================================================
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# =============================================================================
# Constants
# =============================================================================
NAN = "NaN"
RATS = {"dsrc": "DSRC", "pc5": "C-V2X PC5", "5g": "5G SA"}
MODELS = ["lstm", "gru", "rnn"]


# =============================================================================
# Scaler Factory Functions
# =============================================================================
# All scalers use fixed bounds (not data-driven) to ensure consistency
# across training runs and enable proper inverse transformation of predictions.


def create_gps_scaler():
    """
    Create GPS coordinate scaler for latitude/longitude normalization.

    Uses fixed bounds based on Toulouse test area to ensure consistent
    normalization across all datasets.

    Returns:
        Fitted MinMaxScaler for 2D GPS coordinates
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]])


def create_latency_scaler():
    """
    Create latency scaler with expected measurement bounds.

    Bounds (4-50ms) cover typical V2X latency range for all RATs.

    Returns:
        Fitted MinMaxScaler for latency values
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LATENCY], [MAX_LATENCY]])


def create_pdr_scaler():
    """
    Create PDR scaler for packet delivery rate normalization.

    PDR is inherently bounded [0, 1], so scaler is identity transform.

    Returns:
        Fitted MinMaxScaler for PDR values
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_PDR], [MAX_PDR]])


def create_throughput_scaler():
    """
    Create throughput scaler for data rate normalization.

    Returns:
        Fitted MinMaxScaler for throughput values (0-100 Mbps)
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_THROUGHPUT], [MAX_THROUGHPUT]])


def create_sinr_5g_scaler():
    """
    Create 5G SINR scaler for signal quality normalization.

    SINR values from 5G modem are in proprietary units (200-375).

    Returns:
        Fitted MinMaxScaler for 5G SINR values
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_SINR_5G], [MAX_SINR_5G]])


def create_rsrp_5g_scaler():
    """
    Create 5G RSRP scaler for reference signal power normalization.

    RSRP typically ranges from -127 dBm (weak) to -67 dBm (strong).

    Returns:
        Fitted MinMaxScaler for 5G RSRP values
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_RSRP_5G], [MAX_RSRP_5G]])


def create_rsrp_dsrc_scaler():
    """
    Create DSRC RSRP scaler for 802.11p signal power normalization.

    DSRC has wider power range than 5G due to different PHY.

    Returns:
        Fitted MinMaxScaler for DSRC RSRP values
    """
    return MinMaxScaler(feature_range=(0, 1)).fit([[MIN_RSRP_DSRC], [MAX_RSRP_DSRC]])


def create_all_scalers():
    """
    Create dictionary of all scalers for complete preprocessing pipeline.

    GPS coordinates share a single 2D scaler to maintain spatial relationship.
    Each other feature has its own independent scaler.

    Returns:
        Dictionary mapping column names to fitted MinMaxScaler objects
    """
    gps_scaler = create_gps_scaler()
    return {
        'tx_latitude': gps_scaler,
        'tx_longitude': gps_scaler,  # Same scaler as latitude for 2D transform
        'latency_ms': create_latency_scaler(),
        'throughput': create_throughput_scaler(),
        'pdr': create_pdr_scaler(),
        'sinr': create_sinr_5g_scaler(),
        'rsrp': create_rsrp_5g_scaler(),
        'rsrp_1': create_rsrp_dsrc_scaler(),
        'rsrp_2': create_rsrp_dsrc_scaler(),
    }
