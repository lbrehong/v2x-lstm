"""
RAT (Radio Access Technology) selection and analysis module.

This module implements:
- Predictive QoS-based RAT selection algorithm
- Opportunistic (reactive) RAT selection baseline
- Map visualization with Folium
- Performance statistics and histogram generation
- RMSE analysis and comparison tables

Usage:
    python -m selection.rat_selection --input /path/to/csv_folder --model_type all
    python -m selection.rat_selection --input /path/to/csv --mode view
    python -m selection.rat_selection --input /path/to/csv --mode data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from tabulate import tabulate
import folium

from config import (
    MODEL_DIR, OUTPUT_DIR, TIMESTEPS, TARGET_COLS, RATS, MODELS, FEATURE_COLS,
    ALL_RATS, DEFAULT_RATS, LSTM_FALLBACK_RAT, validate_rats,
    PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS,
    create_gps_scaler, create_latency_scaler,
)
from utils import get_latest_model
from learning.model import automatic_train_torch, load_torch_model, predict_torch
from learning.data_preprocessing import preprocess_lstm_input, normalize_features
from selection.file_integration import process_batch

# Initialize scalers for coordinate and latency transformations
gps_scaler = create_gps_scaler()
latency_scalers = {rat: create_latency_scaler(rat) for rat in ("5g", "pc5", "dsrc")}


def grab_gps(df):
    """Extract unique GPS coordinates from dataframe."""
    latitude = df["tx_latitude"]
    longitude = df["tx_longitude"]
    dfr = pd.DataFrame()
    dfr["tx_latitude"] = latitude
    dfr["tx_longitude"] = longitude
    dfr = dfr.drop_duplicates()
    print(f"Loaded {len(dfr)} GPS points")
    return dfr


def build_prediction_windows(rat, df, seq_length=TIMESTEPS):
    """
    Build model input windows from measured data, aligned with training.

    Training pairs rows j..j+seq_length-1 with the target at row j+seq_length,
    so the window for row i is rows i-seq_length .. i-1 (row i is never an input).

    Feature column `c` of FEATURE_COLS[rat] is read from `c_<rat>` when present
    (super_merged naming, e.g. latency_ms_5g), else from `c` (e.g. sinr, tx_latitude).

    Args:
        rat: RAT type identifier ('5g', 'pc5', or 'dsrc')
        df: DataFrame of measurements, one row per time step
        seq_length: Window length (default: TIMESTEPS)

    Returns:
        Tuple of (windows, valid) where windows has shape
        (valid.sum(), seq_length, n_features) and valid is a boolean array over
        df rows. Rows without seq_length prior rows, or whose window contains a
        missing value, are invalid.

    Raises:
        ValueError: If a required feature column is missing from df
    """
    feature_cols = FEATURE_COLS[rat]
    source = {c: f"{c}_{rat}" if f"{c}_{rat}" in df.columns else c for c in feature_cols}
    missing = [src for src in source.values() if src not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {rat} predictions: {missing}")

    raw = pd.DataFrame({c: df[src].to_numpy() for c, src in source.items()})
    features = normalize_features(raw, rat).to_numpy(dtype=float)

    n = len(features)
    valid = np.zeros(n, dtype=bool)
    if n <= seq_length:
        return np.empty((0, seq_length, len(feature_cols))), valid

    # Number of rows with a NaN among the seq_length rows preceding each row
    bad = np.concatenate(([0], np.cumsum(np.isnan(features).any(axis=1))))
    rows = np.arange(seq_length, n)
    valid[rows] = (bad[rows] - bad[rows - seq_length]) == 0

    # Window k covers rows k..k+seq_length-1 and predicts row k+seq_length
    all_windows = np.lib.stride_tricks.sliding_window_view(features, seq_length, axis=0)
    windows = all_windows[np.flatnonzero(valid) - seq_length].transpose(0, 2, 1)
    return np.ascontiguousarray(windows), valid


def get_predictions(model, rat, df):
    """
    Predict latency and PDR for every row of a measurement DataFrame.

    Inputs are built from the measured features of the preceding rows, scaled
    exactly as in training (see build_prediction_windows). Rows without a full
    window of measurements get NaN predictions, which select_best_rat treats as
    the RAT being unavailable.

    Args:
        model: Trained PyTorch model for this RAT
        rat: RAT type identifier ('5g', 'pc5', or 'dsrc')
        df: Super-merged DataFrame (one row per GPS point, in time order)

    Returns:
        Tuple of (latency_ms, pdr) arrays aligned with df rows
    """
    windows, valid = build_prediction_windows(rat, df)
    latency = np.full(len(df), np.nan)
    pdr = np.full(len(df), np.nan)
    print(f"{rat}: predicting {valid.sum()}/{len(df)} rows "
          f"({len(df) - valid.sum()} without a full measurement window)")
    if not valid.any():
        return latency, pdr

    pred_latency, pred_pdr = predict_torch(model, windows)
    pred_latency = np.clip(np.asarray(pred_latency, dtype=float).reshape(-1, 1), 0.0, 1.0)
    latency[valid] = latency_scalers[rat].inverse_transform(pred_latency).ravel()
    pdr[valid] = np.clip(np.asarray(pred_pdr, dtype=float).ravel(), 0.0, 1.0)
    return latency, pdr


def merge_csvs(directory, rats=DEFAULT_RATS):
    """
    Merge base CSV files with matched secondary RAT data, including signal columns.

    Only secondary RATs in `rats` are merged. A selected RAT whose matched CSV
    is missing gets NaN latency/PDR columns, so selection treats it as unavailable.
    """
    rats = validate_rats(rats)
    df_super = pd.read_csv(os.path.join(directory, "super.csv"))

    # 5G: include signal quality columns
    fiveg_path = os.path.join(directory, "matched_5g.csv")
    fiveg_available = pd.read_csv(fiveg_path, nrows=0).columns.tolist()
    fiveg_signal_cols = [c for c in ("sinr", "rsrp") if c in fiveg_available]
    if fiveg_signal_cols:
        df_5g_signal = pd.read_csv(fiveg_path,
                                    usecols=["tx_latitude", "tx_longitude"] + fiveg_signal_cols)
        df_super = df_super.merge(df_5g_signal, on=["tx_latitude", "tx_longitude"], how="left")

    # DSRC (with signal quality columns for feedback loop feature vectors), then PC5
    for rat, signal_cols in (("dsrc", ("rsrp_1", "rsrp_2")), ("pc5", ())):
        if rat not in rats:
            continue
        rat_path = os.path.join(directory, f"matched_{rat}.csv")
        if not os.path.exists(rat_path):
            print(f"  matched_{rat}.csv not found, {rat} marked unavailable")
            df_super[f"latency_ms_{rat}"] = np.nan
            df_super[f"pdr_{rat}"] = np.nan
            continue
        available = pd.read_csv(rat_path, nrows=0).columns.tolist()
        cols = ["tx_latitude", "tx_longitude", "latency_ms", "pdr"] + [c for c in signal_cols if c in available]
        df_rat = pd.read_csv(rat_path, usecols=cols).rename(
            columns={"latency_ms": f"latency_ms_{rat}", "pdr": f"pdr_{rat}"})
        df_super = df_super.merge(df_rat, on=["tx_latitude", "tx_longitude"], how="left")

    df_super.to_csv(os.path.join(directory, "super_merged.csv"), index=False)
    return df_super


def add_predictions(df, directory, model_types=None, output_dir=None, rats=DEFAULT_RATS):
    """Add model predictions to the dataframe.

    Args:
        df: Super-merged DataFrame with tx_latitude/tx_longitude columns.
        directory: Data directory (checked first for final_log files).
        model_types: List of model types to include.
        output_dir: Pipeline output directory (fallback for final_log files).
                    Uses config.OUTPUT_DIR if None — note that the module-level
                    OUTPUT_DIR import is stale when run_pipeline.py overrides it.
        rats: RATs whose predictions are added (default: DEFAULT_RATS).
    """
    import config as _cfg
    fallback_dir = output_dir or _cfg.OUTPUT_DIR

    # Round GPS in the super dataframe for robust matching
    GPS_DECIMALS = 6
    df["_lat_r"] = df["tx_latitude"].round(GPS_DECIMALS)
    df["_lon_r"] = df["tx_longitude"].round(GPS_DECIMALS)

    for model_type in (model_types or MODELS):
        for rat in validate_rats(rats):
            filename = os.path.join(directory, f"final_log_{model_type}_{rat}.csv")

            try:
                df_pred = pd.read_csv(filename, usecols=["latitude", "longitude", "pred_latency", "pred_pdr"])
            except FileNotFoundError:
                df_pred = pd.read_csv(os.path.join(fallback_dir, f"final_log_{model_type}_{rat}.csv"),
                                       usecols=["latitude", "longitude", "pred_latency", "pred_pdr"])
            df_pred.rename(columns={
                "latitude": "tx_latitude",
                "longitude": "tx_longitude",
                "pred_latency": f"pred_latency_ms_{rat}_{model_type}",
                "pred_pdr": f"pred_pdr_{rat}_{model_type}"
            }, inplace=True)

            # Round GPS for matching, then drop duplicates to avoid row explosion
            df_pred["_lat_r"] = df_pred["tx_latitude"].round(GPS_DECIMALS)
            df_pred["_lon_r"] = df_pred["tx_longitude"].round(GPS_DECIMALS)
            pred_cols = [f"pred_latency_ms_{rat}_{model_type}", f"pred_pdr_{rat}_{model_type}"]
            df_pred = df_pred.drop_duplicates(subset=["_lat_r", "_lon_r"], keep="first")

            df = df.merge(df_pred[["_lat_r", "_lon_r"] + pred_cols],
                          on=["_lat_r", "_lon_r"], how="left")

    df.drop(columns=["_lat_r", "_lon_r"], inplace=True)
    return df


def select_best_rat(row, model_type, rats=DEFAULT_RATS):
    """
    Select the optimal RAT based on predicted QoS metrics.

    Implements a reliability-first, latency-optimized selection algorithm:

    Algorithm:
        1. Filter RATs with predicted PDR >= reliability threshold (0.99)
        2. If none qualify, fall back to RAT with highest actual PDR
        3. Among qualified RATs, select the one with lowest predicted latency
        4. Break latency ties (within 1ms) by preferring 5G > PC5 > DSRC

    Args:
        row: DataFrame row containing prediction columns for all RATs
        model_type: Model architecture name (lstm, gru, rnn)
        rats: RATs eligible for selection (default: DEFAULT_RATS)

    Returns:
        String identifier of selected RAT (one of `rats`), the fallback RAT
        (LSTM_FALLBACK_RAT) when the model cannot decide, or 'NaN'
    """
    return _select_best_rat(row, model_type, rats)[0]


def _select_best_rat(row, model_type, rats=DEFAULT_RATS):
    """select_best_rat's decision, plus whether the fallback RAT was used."""
    rats = validate_rats(rats)
    fallback_active = LSTM_FALLBACK_RAT is not None and LSTM_FALLBACK_RAT in rats
    # A RAT without a model or measurements (e.g. no DSRC data) has no columns -> NaN, filtered below
    candidates = [
        (rat, row.get(f"pred_latency_ms_{rat}_{model_type}", np.nan),
         row.get(f"pred_pdr_{rat}_{model_type}", np.nan), row.get(f"pdr_{rat}", np.nan))
        for rat in rats
    ]

    # Filter out options with NaN predictions
    options = [opt for opt in candidates if pd.notna(opt[1]) and pd.notna(opt[2]) and pd.notna(opt[3])]
    if not options:
        # No RAT has both a prediction and a last measurement: the model cannot decide
        return (LSTM_FALLBACK_RAT, True) if fallback_active else ("NaN", False)

    # Filter by PDR reliability threshold
    valid_options = [opt for opt in options if opt[2] >= PDR_RELIABILITY_THRESHOLD]
    if not valid_options:
        # Fallback: filter out unavailable RATs
        keep = [opt for opt in options if opt[3] >= PDR_AVAILABILITY_THRESHOLD]
        if keep:
            best_rat = max(keep, key=lambda x: x[2])[0]
        else:
            fiveg = next((opt for opt in options if opt[0] == "5g"), None)
            if fiveg and fiveg[3] >= PDR_AVAILABILITY_THRESHOLD:
                best_rat = "5g"
            else:
                best_rat = "NaN"
    else:
        # Select lowest latency
        valid_options.sort(key=lambda x: x[1])
        best_latency = valid_options[0][1]
        best_rat = valid_options[0][0]

        # Gather all options within margin of the best, then apply priority
        tied = [opt for opt in valid_options
                if abs(opt[1] - best_latency) < LATENCY_TIE_MARGIN_MS]
        priority = {"5g": 0, "pc5": 1, "dsrc": 2}
        best_rat = min(tied, key=lambda opt: priority.get(opt[0], 99))[0]

    if best_rat == "NaN" and fallback_active and not any(opt[0] == LSTM_FALLBACK_RAT for opt in options):
        # Every usable RAT is unavailable, and the fallback RAT has no prediction to judge it by
        return LSTM_FALLBACK_RAT, True
    return best_rat, False


def lstm_fallback_mask(df, model_type, rats=DEFAULT_RATS):
    """
    Rows where select_best_rat used the fallback RAT (same decision logic, so exact).

    That is: no RAT has both a prediction and a last measured PDR, or every usable
    RAT is unavailable while the fallback RAT has no prediction. Pass the same
    (lagged) frame given to select_best_rat.

    Args:
        df: DataFrame with pred_latency_ms_<rat>_<model>, pred_pdr_<rat>_<model>
            and pdr_<rat> columns
        model_type: Model architecture name (lstm, gru, rnn)
        rats: Active RATs

    Returns:
        Boolean Series (all False when no fallback RAT is configured or active)
    """
    rats = validate_rats(rats)
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    return df.apply(lambda row: _select_best_rat(row, model_type, rats)[1], axis=1).astype(bool)


def lagged_measurements(df, rats=DEFAULT_RATS):
    """
    Copy of df where each row holds the previous row's measured latency/PDR.

    Decisions for row i must only use data observed up to row i-1. Applying
    select_best_rat / opportunistic_best_rat to this frame enforces that; the
    first row has no observation (NaN). Prediction columns are left unchanged.

    Args:
        df: Super-merged DataFrame in time order
        rats: RATs whose latency_ms_<rat> / pdr_<rat> columns are shifted

    Returns:
        New DataFrame with the measurement columns shifted by one row
    """
    lagged = df.copy()
    for rat in validate_rats(rats):
        for col in (f"latency_ms_{rat}", f"pdr_{rat}"):
            if col in lagged.columns:
                lagged[col] = df[col].shift(1)
    return lagged


def opportunistic_best_rat(df, rats=DEFAULT_RATS):
    """
    Select RAT using opportunistic (reactive) algorithm without prediction.

    Baseline algorithm that selects RAT based on current observed metrics
    rather than predictions. Implements a sticky policy to reduce handovers.

    Algorithm:
        1. Filter RATs with PDR > 5%
        2. If currently on 5G and V2X options available, switch to lowest latency
        3. Otherwise, stay on current RAT if still available
        4. Fall back to 5G if available, else mark as unavailable

    Args:
        df: DataFrame with actual latency and PDR columns for the RATs
        rats: RATs eligible for selection (default: DEFAULT_RATS)

    Returns:
        DataFrame with 'Best_RAT_opp' column added
    """
    rats = validate_rats(rats)
    columns = {
        rat: (df[f"latency_ms_{rat}"] if f"latency_ms_{rat}" in df else pd.Series(np.nan, index=df.index),
              df[f"pdr_{rat}"] if f"pdr_{rat}" in df else pd.Series(np.nan, index=df.index))
        for rat in rats
    }
    best_rat_list = []
    previous_rat = "5g"
    for i in range(len(df)):
        options = [
            (rat, columns[rat][0].iat[i], columns[rat][1].iat[i])
            for rat in rats
        ]
        # Filter by PDR threshold (>5%)
        valid_options = [opt for opt in options if opt[2] > 0.05]
        if not valid_options:
            keep = [opt for opt in options if opt[2] > 0.0]
            if keep:
                best_rat = min(keep, key=lambda x: x[1])[0]
                best_rat_list.append(best_rat)
                previous_rat = best_rat
                continue
            else:
                fiveg = next((opt for opt in options if opt[0] == "5g"), None)
                if fiveg and fiveg[2] > PDR_AVAILABILITY_THRESHOLD:
                    best_rat = "5g"
                else:
                    best_rat = "NaN"
                best_rat_list.append(best_rat)
                previous_rat = best_rat
                continue

        if len(valid_options) == 1 and valid_options[0][0] == "5g":
            best_rat = "5g"
        elif previous_rat == "5g" and any(rat[0] != "5g" for rat in valid_options):
            best_rat = min(valid_options, key=lambda x: x[1])[0]
        elif any(rat[0] == previous_rat for rat in valid_options):
            best_rat = previous_rat
        else:
            fiveg = next((opt for opt in options if opt[0] == "5g"), None)
            if fiveg and fiveg[2] > PDR_AVAILABILITY_THRESHOLD:
                best_rat = "5g"
            else:
                best_rat = "NaN"

        best_rat_list.append(best_rat)
        previous_rat = best_rat

    df["Best_RAT_opp"] = best_rat_list
    return df


def process_all(input_dir):
    """Process all RATs and models."""
    for rat in RATS:
        dfl = pd.read_csv(os.path.join(input_dir, f"matched_{rat}.csv"))

        print("___ Starting data preprocessing.")
        print("______ Final set.")
        (X_new, y_new, scalers) = preprocess_lstm_input(dfl, new=True, rat=rat,
                                                         target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("___ Preprocessing complete.")

        for model_type in MODELS:
            model_f = load_torch_model(get_latest_model(model_type, rat))

            print("____________________________________________________")
            print("Automatic retraining.")
            automatic_train_torch(model_f, X_new, y_new, 32, 200, 0.15,
                                  os.path.join(OUTPUT_DIR, f"final_log_{model_type}_{rat}.csv"), rat, model_type)

            print(f"{rat}: {model_type} predictions done.")


def process_file(input_csv, model_type, output_csv, input_dir):
    """Process a single file with specified model type."""
    dfl = pd.read_csv(os.path.join(input_dir, input_csv))
    df_gps = grab_gps(dfl)
    df = pd.DataFrame()
    df["tx_latitude"] = df_gps["tx_latitude"]
    df["tx_longitude"] = df_gps["tx_longitude"]

    print("___ Starting data preprocessing.")
    print("______ Training set.")
    (X_5g, y_5g, scalers) = preprocess_lstm_input(df, new=True, rat="5g",
                                                   target_cols=TARGET_COLS, seq_length=TIMESTEPS)
    (X_pc5, y_pc5, scalers) = preprocess_lstm_input(df, new=True, rat="pc5",
                                                     target_cols=TARGET_COLS, seq_length=TIMESTEPS)
    (X_dsrc, y_dsrc, scalers) = preprocess_lstm_input(df, new=True, rat="dsrc",
                                                       target_cols=TARGET_COLS, seq_length=TIMESTEPS)

    model_dsrc = load_torch_model(get_latest_model(model_type, "dsrc"))
    model_cv2x = load_torch_model(get_latest_model(model_type, "pc5"))
    model_5g = load_torch_model(get_latest_model(model_type, "5g"))

    print("GPS Data shape:", df.shape)
    print("First few GPS entries:\n", df[:5])

    df_pred = pd.DataFrame()
    df_pred[f"pred_latency_ms_dsrc_{model_type}"], df_pred[f"pred_pdr_dsrc_{model_type}"] = get_predictions(model_dsrc, "dsrc", df)
    df_pred[f"pred_latency_ms_pc5_{model_type}"], df_pred[f"pred_pdr_pc5_{model_type}"] = get_predictions(model_cv2x, "pc5", df)
    df_pred[f"pred_latency_ms_5g_{model_type}"], df_pred[f"pred_pdr_5g_{model_type}"] = get_predictions(model_5g, "5g", df)

    # Add actual PDR columns (required by select_best_rat fallback)
    for rat in ("dsrc", "pc5", "5g"):
        df_pred[f"pdr_{rat}"] = df_pred[f"pred_pdr_{rat}_{model_type}"]

    df[f"Best_RAT_{model_type}"] = df_pred.apply(lambda row: select_best_rat(row, model_type), axis=1)

    output_path = os.path.join(input_dir, f"output_{input_csv}")
    df.to_csv(output_path, index=False)
    print(f"Processed file saved to {output_path}")


def visualize_rat_map(output_csv, map_output="rat_map", output_dir=OUTPUT_DIR):
    """Generate map visualization of RAT selection."""
    color_map = {"dsrc": "blue", "pc5": "orange", "5g": "green"}
    # Use local list to avoid mutating global MODELS
    models_with_opp = MODELS + ["opp"]

    for model_type in models_with_opp:
        df = pd.read_csv(output_csv)
        start_location = [df.iloc[0]["tx_latitude"], df.iloc[0]["tx_longitude"]]
        m = folium.Map(location=start_location, zoom_start=14, tiles="OpenStreetMap")

        for i in range(len(df) - 1):
            lat1, lon1, rat1 = df.iloc[i][["tx_latitude", "tx_longitude", f"Best_RAT_{model_type}"]]
            lat2, lon2, rat2 = df.iloc[i + 1][["tx_latitude", "tx_longitude", f"Best_RAT_{model_type}"]]
            folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                            color=color_map.get(rat1, "gray"),
                            weight=5,
                            opacity=0.8).add_to(m)

        m.save(os.path.join(output_dir, f"{map_output}_{model_type}.html"))
        print(f"Map saved to {map_output}_{model_type}.html")


def get_latencies(df, scheme_column):
    """Extract latencies based on selected RAT for each scheme."""
    return df.apply(lambda row: row[f"latency_ms_{row[scheme_column].lower()}"]
                    if pd.notna(row[scheme_column]) else None, axis=1)


def get_pdr(df, scheme_column):
    """Extract PDR based on selected RAT for each scheme."""
    return df.apply(lambda row: row[f"pdr_{row[scheme_column].lower()}"] * 100
                    if pd.notna(row[scheme_column]) else None, axis=1)


def make_histogram_latency(input_csv, output_dir=OUTPUT_DIR):
    """Generate latency histogram."""
    df = pd.read_csv(input_csv)

    bins = [0, 10, 20, 50, float("inf")]
    labels = ["<10ms", "10-20ms", "20-50ms", ">50ms"]

    latency_lstm = get_latencies(df, "Best_RAT_lstm")
    latency_gru = get_latencies(df, "Best_RAT_gru")
    latency_rnn = get_latencies(df, "Best_RAT_rnn")
    latency_opportunistic = get_latencies(df, "Best_RAT_opp")

    category_data = pd.DataFrame({
        "LSTM": pd.cut(latency_lstm, bins=bins, labels=labels),
        "GRU": pd.cut(latency_gru, bins=bins, labels=labels),
        "RNN": pd.cut(latency_rnn, bins=bins, labels=labels),
        "No pQoS": pd.cut(latency_opportunistic, bins=bins, labels=labels),
    })

    hist_data = category_data.apply(lambda x: x.value_counts(normalize=True) * 100)

    ax = hist_data.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("Latency Distribution by Scheme")
    plt.xlabel("Latency Category")
    plt.ylabel("Percentage of Total Transmissions")
    plt.xticks(rotation=0)
    plt.legend(title="Selection Scheme")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(output_dir, "latency_histogram.png"))
    plt.show()


def make_histogram_pdr(input_csv, output_dir=OUTPUT_DIR):
    """Generate PDR histogram."""
    df = pd.read_csv(input_csv)

    bins = [0.0, 95.0, 99.0, 99.9, 100.0]
    labels = ["<95%", "99-95%", "99.9-99%", ">99.9%"]

    pdr_lstm = get_pdr(df, "Best_RAT_lstm")
    pdr_gru = get_pdr(df, "Best_RAT_gru")
    pdr_rnn = get_pdr(df, "Best_RAT_rnn")
    pdr_opportunistic = get_pdr(df, "Best_RAT_opp")

    category_data = pd.DataFrame({
        "LSTM": pd.cut(pdr_lstm, bins=bins, labels=labels),
        "GRU": pd.cut(pdr_gru, bins=bins, labels=labels),
        "RNN": pd.cut(pdr_rnn, bins=bins, labels=labels),
        "No pQoS": pd.cut(pdr_opportunistic, bins=bins, labels=labels),
    })

    hist_data = category_data.apply(lambda x: x.value_counts(normalize=True) * 100).reindex(labels[::-1])
    count_data = category_data.apply(lambda x: x.value_counts()).reindex(labels[::-1])
    ci_ranges = count_data.map(lambda n: (st.t.interval(0.95, df=n - 1, loc=n, scale=np.sqrt(n))[1] - n) if n > 1 else 0)
    ci_ranges = (ci_ranges / count_data.sum()) * 100

    fig, ax = plt.subplots(figsize=(20, 12))
    hist_data.plot(kind="bar", yerr=ci_ranges, capsize=5, ax=ax, width=0.8, error_kw={'elinewidth': 2, 'alpha': 0.6})

    plt.title("PDR Distribution by Model Type", fontsize=28)
    plt.xlabel("PDR Category", fontsize=24)
    plt.ylabel("Percentage of Total Transmissions (%)", fontsize=28)
    plt.xticks(rotation=0, fontsize=22)
    plt.yticks(np.arange(0, 101, 10), fontsize=22)
    plt.legend(title="Model Type", fontsize=22, title_fontsize=24)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(output_dir, "pdr_histogram.png"))
    plt.show()


def mean_ci(series, confidence=0.95):
    """Calculate mean and confidence interval."""
    series = series.dropna()
    mean = np.mean(series)
    if len(series) > 1:
        ci = st.t.interval(confidence, len(series) - 1, loc=mean, scale=st.sem(series))
        ci_range = ci[1] - mean
    else:
        ci_range = 0
    return mean, ci_range


def get_metric(df, scheme_column, metric):
    """Extract metric based on selected RAT."""
    return df.apply(lambda row: row[f"{metric}_{row[scheme_column].lower()}"]
                    if pd.notna(row[scheme_column]) else None, axis=1)


def make_table(input_csv, output_dir=OUTPUT_DIR, rats=DEFAULT_RATS):
    """Generate summary statistics table."""
    df = pd.read_csv(input_csv)
    rats = validate_rats(rats)

    summary_data = {}
    schemes = ["Best_RAT_lstm", "Best_RAT_gru", "Best_RAT_rnn", "Best_RAT_opp"]

    for scheme in schemes:
        scheme_name = scheme.replace("Best_RAT_", "").upper()

        pdr_values = get_pdr(df, scheme).copy()
        latency_values = get_latencies(df, scheme).copy()

        avg_pdr, ci_pdr = mean_ci(pdr_values)
        avg_latency, ci_latency = mean_ci(latency_values)
        max_latency = np.max(latency_values.dropna()) if not latency_values.dropna().empty else np.nan

        total_messages = len(df)

        per_rat_latency = [mean_ci(get_latencies(df[df[scheme] == rat], scheme)) for rat in rats]
        per_rat_usage = [((df[scheme] == rat).sum() / total_messages * 100, 0) for rat in rats]

        summary_data[scheme_name] = [
            (avg_pdr, ci_pdr),
            (avg_latency, ci_latency),
            *per_rat_latency,
            (max_latency, 0),
            *per_rat_usage,
        ]

    summary_df = pd.DataFrame(summary_data, index=[
        "Avg PDR", "Avg Latency",
        *[f"Avg {rat.upper()} Latency" for rat in rats],
        "Max Latency",
        *[f"{rat.upper()} Usage (%)" for rat in rats],
    ])

    summary_df = summary_df.map(lambda x: f"{x[0]:.3f} ± {x[1]:.3f}" if isinstance(x, tuple) else x)

    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
    summary_df.to_csv(os.path.join(output_dir, "best_perf_table.csv"), index=False)


def make_rmse_table(input_csv=OUTPUT_DIR, output_dir=OUTPUT_DIR, rats=DEFAULT_RATS):
    """Generate RMSE summary table."""
    rats = validate_rats(rats)
    rmse_summary = {}

    for model_type in MODELS:
        scheme_data = []
        for rat in rats:
            df = pd.read_csv(os.path.join(input_csv, f"final_log_{model_type}_{rat}.csv"))
            # MAE: mean of absolute errors
            latency_mae, ci_latency_mae = mean_ci(df["mae_latency"])
            pdr_mae, ci_pdr_mae = mean_ci(df["mae_pdr"])
            # RMSE: mean of per-batch RMSE values
            latency_rmse = df["rmse_latency"].mean()
            pdr_rmse = df["rmse_pdr"].mean()

            scheme_data.append((
                f"MAE: {latency_mae:.3f}±{ci_latency_mae:.3f} | RMSE: {latency_rmse:.3f}",
                f"MAE: {pdr_mae:.3f}±{ci_pdr_mae:.3f} | RMSE: {pdr_rmse:.3f}",
            ))

        rmse_summary[model_type] = scheme_data

    rmse_summary_df = pd.DataFrame(rmse_summary, index=[rat.upper() for rat in rats])
    rmse_summary_df.columns = ["LSTM", "GRU", "RNN"]
    rmse_summary_df.index.name = "RAT"
    rmse_summary_df.to_csv(os.path.join(output_dir, "best_rmse_table.csv"))


def make_rmse_plot(input_csv=OUTPUT_DIR, output_dir=OUTPUT_DIR, rats=DEFAULT_RATS):
    """Generate RMSE plots."""
    rats = validate_rats(rats)
    window_size = 250
    titles = ["LSTM", "GRU", "SimpleRNN"]
    colors = {"dsrc": "blue", "pc5": "orange", "5g": "green"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for i, model_type in enumerate(MODELS):
        ax_lat = axes[0, i]
        ax_pdr = axes[1, i]
        for rat in rats:
            df = pd.read_csv(os.path.join(input_csv, f"final_log_{model_type}_{rat}.csv"))

            latency_mae_ma = df["mae_latency"].rolling(window=window_size, min_periods=1).mean()
            ax_lat.plot(latency_mae_ma, label=f"Latency MAE {rat.upper()}", linestyle="-", color=colors[rat])

            pdr_mae_ma = df["mae_pdr"].rolling(window=window_size, min_periods=1).mean()
            ax_pdr.plot(pdr_mae_ma, label=f"PDR MAE {rat.upper()}", linestyle="-", color=colors[rat])

        ax_lat.set_title(titles[i])
        ax_lat.set_xlabel("Message Index")
        ax_lat.set_ylabel("Latency MAE")
        ax_lat.set_ylim(0, 15)
        ax_lat.grid(True, linestyle="--", alpha=0.5)
        ax_lat.legend()

        ax_pdr.set_title(titles[i])
        ax_pdr.set_xlabel("Message Index")
        ax_pdr.set_ylabel("PDR MAE")
        ax_pdr.set_ylim(0, 1)
        ax_pdr.grid(True, linestyle="--", alpha=0.5)
        ax_pdr.legend()

    plt.tight_layout()
    plt.suptitle("Latency & PDR MAE Moving Averages per Scheme", fontsize=14, y=1.05)
    plt.savefig(os.path.join(output_dir, "rmse_pred_plot.png"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best RAT for given GPS coordinates")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV folder")
    parser.add_argument('--mode', type=str, help="Mode: empty for calculate, 'test' for GPS data, 'view' for visualize, 'data' for statistics, 'api_batch' for queue simulator integration")
    parser.add_argument('--model_type', type=str, help="RNN model type (lstm, gru, rnn, or all)")
    parser.add_argument('--output', type=str, help="Output path for api_batch mode")
    parser.add_argument('--rats', nargs='+', default=list(DEFAULT_RATS), choices=ALL_RATS,
                        help=f"RATs eligible for selection (default: {' '.join(DEFAULT_RATS)})")
    args = parser.parse_args()

    INPUT = args.input
    MODEL_TYPE = args.model_type
    MODE = args.mode
    ACTIVE_RATS = validate_rats(args.rats)

    if not MODE and MODEL_TYPE:
        if os.path.isdir(INPUT):
            # Directory mode: process matched_*.csv files
            model_types = MODELS if MODEL_TYPE == "all" else [MODEL_TYPE]

            print("__________________________________")
            print("____________ LET'S GO ____________")
            print("__________________________________")
            print(f"_ Processing all input CSVs with model type(s): {model_types}")

            # Build super_merged from matched CSVs
            print("_ Merging into the super-CSV.")
            super_df = merge_csvs(INPUT, rats=ACTIVE_RATS)

            # Predict from the measured history in super_merged — no intermediate files
            for mt in model_types:
                for rat in ACTIVE_RATS:
                    model_path = get_latest_model(mt, rat)
                    if model_path is None:
                        print(f"  Warning: no {mt} model found for {rat}, skipping")
                        continue
                    model_f = load_torch_model(model_path)
                    latency, pdr = get_predictions(model_f, rat, super_df)
                    super_df[f"pred_latency_ms_{rat}_{mt}"] = latency
                    super_df[f"pred_pdr_{rat}_{mt}"] = pdr
                    print(f"  {rat}: {mt} predictions done.")

            print("_ Predictions added.")
            print("_ Sending to the selection algorithm.")
            # Decide row i from row i-1's measurements (no lookahead)
            observed = lagged_measurements(super_df, ACTIVE_RATS)
            for mt in model_types:
                super_df[f"Best_RAT_{mt}"] = observed.apply(select_best_rat, args=(mt, ACTIVE_RATS), axis=1)
                super_df[f"Fallback_{mt}"] = lstm_fallback_mask(observed, mt, ACTIVE_RATS)
            print("_ Adding opportunistic algorithm.")
            super_df["Best_RAT_opp"] = opportunistic_best_rat(observed, rats=ACTIVE_RATS)["Best_RAT_opp"]
            print("_ Algorithms done.")
            out_path = os.path.join(INPUT, "bestRAT_super.csv")
            print(f"_ Saving. {out_path}")
            super_df.to_csv(out_path, index=False)
            print("_ All done.")
        else:
            # Single CSV file mode
            process_file(INPUT, MODEL_TYPE, os.path.join(os.path.dirname(INPUT), "selection_results.csv"), INPUT)

    elif MODE == "view":
        visualize_rat_map(INPUT, output_dir=os.path.dirname(INPUT) or OUTPUT_DIR)

    elif MODE == "data":
        make_histogram_latency(INPUT)
        make_histogram_pdr(INPUT)
        make_rmse_plot(rats=ACTIVE_RATS)
        make_rmse_table(rats=ACTIVE_RATS)
        make_table(INPUT, rats=ACTIVE_RATS)

    elif MODE == "test":
        df_gps = grab_gps(INPUT)
        latitude = df_gps["tx_latitude"]
        longitude = df_gps["tx_longitude"]

        map_center = [latitude.mean(), longitude.mean()]
        print(map_center)

        m = folium.Map(
            location=map_center,
            zoom_start=14,
            tiles="Esri.WorldImagery",
            attr="Esri"
        )

        for lat, lon in zip(latitude, longitude):
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
            ).add_to(m)

        m.save(os.path.join(OUTPUT_DIR, "gps_map_pc5_01.html"))

    elif MODE == "api_batch":
        # Queue simulator integration mode
        # Outputs rat_decisions.csv with predictions for all RATs
        output_path = args.output or os.path.join(OUTPUT_DIR, "rat_decisions.csv")
        model_type = MODEL_TYPE or "lstm"
        print(f"Processing {INPUT} with {model_type} model for queue simulator integration")
        process_batch(INPUT, output_path, model_type)
        print(f"RAT decisions saved to {output_path}")

    elif MODE:
        raise ValueError("Invalid mode specified. test, view, data, api_batch and <empty> are valid options.")

    else:
        raise ValueError("No model type specified. all, lstm, gru, rnn are valid options.")
