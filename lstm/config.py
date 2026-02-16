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
MIN_LATENCY, MAX_LATENCY = 0, 300  # in ms (covers 5G p95≈391, DSRC 0-19, PC5 5-93)
MIN_PDR, MAX_PDR = 0.0, 1.0
MIN_THROUGHPUT, MAX_THROUGHPUT = 0, 100  # in Mbps
MIN_SINR_5G, MAX_SINR_5G = -10, 45  # standard 3GPP SINR in dB
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
TX_INTERVAL_MS = {
    "5g": 2000,   # 5G SA ping interval (0.5 Hz, measured from df_ping.json)
    "pc5": 100,   # C-V2X PC5 CAM rate (10 Hz)
    "dsrc": 100,  # DSRC BSM rate (10 Hz)
}
DEFAULT_TX_INTERVAL_MS = 100  # Fallback for unspecified RATs


def get_tx_interval_ms(rat) -> int:
    """Get TX interval in ms for a RAT (string key or RATType enum)."""
    key = rat.value if hasattr(rat, "value") else str(rat)
    return TX_INTERVAL_MS.get(key, DEFAULT_TX_INTERVAL_MS)
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
    "dsrc": {"min": 1024, "max": 4096, "levels": 4},
    "pc5": {"min": 1024, "max": 4096, "levels": 4},
    "5g": {"min": 1024, "max": 4096, "levels": 4},
}

# DTMC (Discrete Time Markov Chain) Packet Sizing Parameters
# Packet sizes: 1024 -> 2048 -> 3072 -> 4096 bytes (increments of 1024)
DTMC_PACKET_SIZES = [1024, 2048, 3072, 4096]  # Explicit size levels
DTMC_THRESHOLD_HIGH = 0.99  # PDR above this -> increase packet size
DTMC_THRESHOLD_LOW = 0.95   # PDR below this -> decrease packet size
DTMC_WINDOW_SECONDS = 1.0   # Moving window for PDR averaging (seconds)
DTMC_P_UP = 1.0             # Deterministic transitions (no randomness)
DTMC_P_DOWN = 1.0           # Deterministic transitions (no randomness)

# =============================================================================
# PHY-Layer Configuration (3GPP/IEEE Standards)
# =============================================================================
# Unified OFDM capacity model for realistic simulation.
#
# All RATs use: capacity = total_data_subcarriers × Nsym × Rmod × CR
#
# For 5G/PC5 (RB-based):   total_data_subcarriers = NSC × NRB
#   NSC   = Subcarriers per RB (12 for LTE/NR)
#   Nsym  = Symbols per subframe (14 for normal CP)
#   NRB   = Resource blocks (total or per subchannel × num subchannels)
# For DSRC (802.11p):       total_data_subcarriers = 52 (direct)
#   Nsym  = 125 symbols/ms (8 μs OFDM symbol duration)
#
# Common:
#   Rmod  = Bits per symbol (modulation order: QPSK=2, 16QAM=4, 64QAM=6)
#   CR    = Channel coding rate (0.5 - 0.9)

RAT_PHY_CONFIG = {
    # 5G NR (FR1, 15 kHz SCS)
    # 1 RB = 12 subcarriers × 14 OFDM symbols per slot
    # 1 slot = 1ms (15 kHz SCS), 1 frame = 10ms = 10 slots
    # 20 MHz → 106 RBs
    "5g": {
        # RB-based capacity formula components
        "n_subcarriers": 12,             # NSC: fixed by NR spec
        "n_symbols": 14,                 # Nsym: symbols per slot (normal CP)
        "n_rbs": 106,                    # NRB: 20 MHz bandwidth
        "modulation_order": 6,           # Rmod: 64QAM (typical for good channel)
        "coding_rate": 0.66,             # CR: typical for 64QAM MCS
        # Timing
        "slot_duration_ms": 1,           # 1 slot = 1ms for 15 kHz SCS
        "base_latency_ms": 15.0,         # Typical E2E latency
        # Queue (2 TX intervals buffer)
        "queue_multiplier": 2,
        # Contention parameters
        "scheduling_delay_per_ue_ms": 0.5,  # Per-UE scheduling overhead
    },
    # C-V2X PC5 Mode 4 (LTE Sidelink)
    # 1 RB = 12 subcarriers × 14 symbols per subframe (2 slots × 7 symbols)
    # 1 subframe = 1ms, 10 MHz → 50 RBs available
    # Subchannel-based allocation for sidelink
    "pc5": {
        # RB-based capacity formula components
        "n_subcarriers": 12,             # NSC: fixed by LTE spec
        "n_symbols": 14,                 # Nsym: symbols per subframe (2 slots × 7)
        "n_rbs_per_subchannel": 10,      # RBs per subchannel (configurable)
        "n_subchannels": 1,              # Subchannels per vehicle (1 typical)
        "n_subchannels_total": 5,        # Total subchannel pool (50 RBs / 10 per subchannel)
        "modulation_order": 2,           # Rmod: QPSK (default, conservative for V2X)
        "coding_rate": 0.5,              # CR: 0.5 (high reliability mode)
        # Timing
        "subframe_duration_ms": 1,       # 1 subframe = 1ms
        "base_latency_ms": 8.0,          # Typical E2E latency
        # Queue (2 TX intervals buffer)
        "queue_multiplier": 2,
    },
    # DSRC 802.11p
    # 10 MHz OFDM channel, 64 subcarriers (52 data, 4 pilot, 4 null/guard)
    # OFDM symbol = 8 μs (6.4 μs useful + 1.6 μs GI) → 125 symbols/ms
    # Unified OFDM formula: capacity = n_data_subcarriers × n_symbols × Rmod × CR
    # QPSK 1/2: 52 × 125 × 2 × 0.5 = 6500 bits/ms ≈ 6.5 Mbps
    "dsrc": {
        # OFDM-level capacity (analogous to RB formula for 5G/PC5)
        "n_data_subcarriers": 52,        # 52 of 64 subcarriers carry data
        "n_symbols": 125,                # symbols per ms (8 μs symbol duration)
        "modulation_order": 2,           # Rmod: QPSK
        "coding_rate": 0.5,              # CR: 1/2 rate
        # Timing
        "base_latency_ms": 5.0,          # Typical E2E latency (contention-based)
        # Contention parameters
        "contention_window_min": 15,     # 802.11p CWmin
        "contention_window_max": 1023,   # 802.11p CWmax
        "slot_time_us": 13,              # 802.11p slot time (μs)
        # Queue (2 TX intervals buffer)
        "queue_multiplier": 2,
    },
}

# MCS/CQI Reference Table (for adaptive modulation - future use)
# CQI | Modulation | Rmod | Typical CR | Use Case
# 1-6 | QPSK       | 2    | 0.08-0.6   | Low SNR, high reliability
# 7-9 | 16QAM      | 4    | 0.37-0.6   | Medium SNR
# 10-15| 64QAM     | 6    | 0.46-0.93  | High SNR, max throughput

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

    SINR in standard 3GPP dB units (-10 to 45 dB).

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
