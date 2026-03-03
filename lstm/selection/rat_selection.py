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
from keras.models import load_model

from config import (
    MODEL_DIR, OUTPUT_DIR, TIMESTEPS, TARGET_COLS, RATS, MODELS,
    PDR_RELIABILITY_THRESHOLD, PDR_AVAILABILITY_THRESHOLD, LATENCY_TIE_MARGIN_MS,
    create_gps_scaler, create_latency_scaler,
)
from utils import get_latest_model
from learning.model import rmse, automatic_train
from learning.data_preprocessing import preprocess_lstm_input
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


def get_predictions(model, rat, gps_data):
    """Get latency and PDR predictions for GPS data."""
    print(f"Prediction over {len(gps_data)} GPS points")

    gps_data_scaled = gps_scaler.transform(gps_data)
    sequence_length = 10
    num_features = 4 if rat == "pc5" else 6

    input_sequences = []

    for i in range(len(gps_data_scaled)):
        past_points = gps_data_scaled[max(0, i - sequence_length + 1):i + 1]
        past_points_full = np.hstack((past_points, np.zeros((past_points.shape[0], num_features - past_points.shape[1]))))

        if len(past_points) < sequence_length:
            padding = np.zeros((sequence_length - len(past_points), num_features))
            sequence = np.vstack((padding, past_points_full))
        else:
            sequence = past_points_full

        input_sequences.append(sequence)

    input_sequences = np.array(input_sequences)

    print("Input Sequences Shape:", input_sequences.shape)
    if input_sequences.shape[0] == 0:
        raise ValueError("Error: No input sequences generated. Check GPS preprocessing.")

    predictions = model.predict(input_sequences)
    predictions = np.array(predictions)

    if predictions.shape == (2, len(gps_data), 1):
        predictions = predictions.reshape(len(gps_data), 2)

    print("Processed Predictions Shape:", predictions.shape)

    pred_latency, pred_pdr = predictions[:, 0].flatten(), predictions[:, 1].flatten()
    pred_latency = np.clip(pred_latency, 0.0, 1.0)
    latency = latency_scalers[rat].inverse_transform(np.array(pred_latency).reshape(-1, 1)).flatten()
    pdr = np.clip(pred_pdr, 0.0, 1.0)

    print("First few latency predictions:", latency[:5])
    print("First few PDR predictions:", pdr[:5])

    return latency, pdr


def merge_csvs(directory):
    """Merge base CSV files with matched DSRC and PC5 data, including signal columns."""
    df_super = pd.read_csv(os.path.join(directory, "super.csv"))

    # DSRC: include signal quality columns for feedback loop feature vectors
    dsrc_cols = ["tx_latitude", "tx_longitude", "latency_ms", "pdr"]
    dsrc_path = os.path.join(directory, "matched_dsrc.csv")
    dsrc_available = pd.read_csv(dsrc_path, nrows=0).columns.tolist()
    for col in ("rsrp_1", "rsrp_2"):
        if col in dsrc_available:
            dsrc_cols.append(col)
    df_base_dsrc = pd.read_csv(dsrc_path, usecols=dsrc_cols)
    df_base_dsrc.rename(columns={"latency_ms": "latency_ms_dsrc", "pdr": "pdr_dsrc"}, inplace=True)

    # PC5
    df_base_pc5 = pd.read_csv(os.path.join(directory, "matched_pc5.csv"),
                               usecols=["tx_latitude", "tx_longitude", "latency_ms", "pdr"])
    df_base_pc5.rename(columns={"latency_ms": "latency_ms_pc5", "pdr": "pdr_pc5"}, inplace=True)

    # 5G: include signal quality columns
    fiveg_path = os.path.join(directory, "matched_5g.csv")
    fiveg_available = pd.read_csv(fiveg_path, nrows=0).columns.tolist()
    fiveg_signal_cols = [c for c in ("sinr", "rsrp") if c in fiveg_available]
    if fiveg_signal_cols:
        df_5g_signal = pd.read_csv(fiveg_path,
                                    usecols=["tx_latitude", "tx_longitude"] + fiveg_signal_cols)
        df_super = df_super.merge(df_5g_signal, on=["tx_latitude", "tx_longitude"], how="left")

    df_super = df_super.merge(df_base_dsrc, on=["tx_latitude", "tx_longitude"], how="left")
    df_super = df_super.merge(df_base_pc5, on=["tx_latitude", "tx_longitude"], how="left")

    df_super.to_csv(os.path.join(directory, "super_merged.csv"), index=False)
    return df_super


def add_predictions(df, directory, model_types=None, output_dir=None):
    """Add model predictions to the dataframe.

    Args:
        df: Super-merged DataFrame with tx_latitude/tx_longitude columns.
        directory: Data directory (checked first for final_log files).
        model_types: List of model types to include.
        output_dir: Pipeline output directory (fallback for final_log files).
                    Uses config.OUTPUT_DIR if None — note that the module-level
                    OUTPUT_DIR import is stale when run_pipeline.py overrides it.
    """
    import config as _cfg
    fallback_dir = output_dir or _cfg.OUTPUT_DIR

    # Round GPS in the super dataframe for robust matching
    GPS_DECIMALS = 6
    df["_lat_r"] = df["tx_latitude"].round(GPS_DECIMALS)
    df["_lon_r"] = df["tx_longitude"].round(GPS_DECIMALS)

    for model_type in (model_types or MODELS):
        for rat in RATS:
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


def select_best_rat(row, model_type):
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

    Returns:
        String identifier of selected RAT: 'dsrc', 'pc5', '5g', or 'NaN'
    """
    options = [
        ("dsrc", row[f"pred_latency_ms_dsrc_{model_type}"], row[f"pred_pdr_dsrc_{model_type}"], row["pdr_dsrc"]),
        ("pc5", row[f"pred_latency_ms_pc5_{model_type}"], row[f"pred_pdr_pc5_{model_type}"], row["pdr_pc5"]),
        ("5g", row[f"pred_latency_ms_5g_{model_type}"], row[f"pred_pdr_5g_{model_type}"], row["pdr_5g"]),
    ]

    # Filter out options with NaN predictions
    options = [opt for opt in options if pd.notna(opt[1]) and pd.notna(opt[2]) and pd.notna(opt[3])]
    if not options:
        return "NaN"

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

    return best_rat


def opportunistic_best_rat(df):
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
        df: DataFrame with actual latency and PDR columns for all RATs

    Returns:
        DataFrame with 'Best_RAT_opp' column added
    """
    best_rat_list = []
    previous_rat = "5g"
    for row in df.itertuples():
        options = [
            ("dsrc", row.latency_ms_dsrc, row.pdr_dsrc),
            ("pc5", row.latency_ms_pc5, row.pdr_pc5),
            ("5g", row.latency_ms_5g, row.pdr_5g),
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
        elif previous_rat == "5g" and any(rat[0] in ["dsrc", "pc5"] for rat in valid_options):
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
            model_f = load_model(get_latest_model(model_type, rat), custom_objects={'rmse': rmse})

            print("____________________________________________________")
            print("Automatic retraining.")
            automatic_train(model_f, X_new, y_new, 32, 200, 0.15,
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

    model_dsrc = load_model(get_latest_model(model_type, "dsrc"), custom_objects={'rmse': rmse})
    model_cv2x = load_model(get_latest_model(model_type, "pc5"), custom_objects={'rmse': rmse})
    model_5g = load_model(get_latest_model(model_type, "5g"), custom_objects={'rmse': rmse})

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


def make_table(input_csv, output_dir=OUTPUT_DIR):
    """Generate summary statistics table."""
    df = pd.read_csv(input_csv)

    summary_data = {}
    schemes = ["Best_RAT_lstm", "Best_RAT_gru", "Best_RAT_rnn", "Best_RAT_opp"]

    for scheme in schemes:
        scheme_name = scheme.replace("Best_RAT_", "").upper()

        pdr_values = get_pdr(df, scheme).copy()
        latency_values = get_latencies(df, scheme).copy()

        avg_pdr, ci_pdr = mean_ci(pdr_values)
        avg_latency, ci_latency = mean_ci(latency_values)
        max_latency = np.max(latency_values.dropna()) if not latency_values.dropna().empty else np.nan

        rat_latencies = {
            "dsrc": get_latencies(df[df[scheme] == "dsrc"], scheme),
            "pc5": get_latencies(df[df[scheme] == "pc5"], scheme),
            "5g": get_latencies(df[df[scheme] == "5g"], scheme),
        }

        avg_dsrc_latency, ci_dsrc = mean_ci(rat_latencies["dsrc"])
        avg_pc5_latency, ci_pc5 = mean_ci(rat_latencies["pc5"])
        avg_5g_latency, ci_5g = mean_ci(rat_latencies["5g"])

        total_messages = len(df)

        pct_dsrc = (df[scheme] == "dsrc").sum() / total_messages * 100
        pct_pc5 = (df[scheme] == "pc5").sum() / total_messages * 100
        pct_5g = (df[scheme] == "5g").sum() / total_messages * 100

        summary_data[scheme_name] = [
            (avg_pdr, ci_pdr),
            (avg_latency, ci_latency),
            (avg_dsrc_latency, ci_dsrc),
            (avg_pc5_latency, ci_pc5),
            (avg_5g_latency, ci_5g),
            (max_latency, 0),
            (pct_dsrc, 0),
            (pct_pc5, 0),
            (pct_5g, 0)
        ]

    summary_df = pd.DataFrame(summary_data, index=[
        "Avg PDR", "Avg Latency", "Avg DSRC Latency", "Avg PC5 Latency", "Avg 5G Latency",
        "Max Latency", "DSRC Usage (%)", "PC5 Usage (%)", "5G Usage (%)"
    ])

    summary_df = summary_df.map(lambda x: f"{x[0]:.3f} ± {x[1]:.3f}" if isinstance(x, tuple) else x)

    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
    summary_df.to_csv(os.path.join(output_dir, "best_perf_table.csv"), index=False)


def make_rmse_table(input_csv=OUTPUT_DIR, output_dir=OUTPUT_DIR):
    """Generate RMSE summary table."""
    rmse_summary = {}

    for model_type in MODELS:
        scheme_data = []
        for rat in RATS:
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

    rmse_summary_df = pd.DataFrame(rmse_summary, index=["DSRC", "PC5", "5G"])
    rmse_summary_df.columns = ["LSTM", "GRU", "RNN"]
    rmse_summary_df.index.name = "RAT"
    rmse_summary_df.to_csv(os.path.join(output_dir, "best_rmse_table.csv"))


def make_rmse_plot(input_csv=OUTPUT_DIR, output_dir=OUTPUT_DIR):
    """Generate RMSE plots."""
    window_size = 250
    titles = ["LSTM", "GRU", "SimpleRNN"]
    colors = {"dsrc": "blue", "pc5": "orange", "5g": "green"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    for i, model_type in enumerate(MODELS):
        ax_lat = axes[0, i]
        ax_pdr = axes[1, i]
        for rat in RATS:
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
    args = parser.parse_args()

    INPUT = args.input
    MODEL_TYPE = args.model_type
    MODE = args.mode

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
            super_df = merge_csvs(INPUT)

            # Predict directly on super_merged GPS — no intermediate files
            gps_data = super_df[["tx_latitude", "tx_longitude"]].copy()

            for mt in model_types:
                for rat in RATS:
                    model_path = get_latest_model(mt, rat)
                    if model_path is None:
                        print(f"  Warning: no {mt} model found for {rat}, skipping")
                        continue
                    model_f = load_model(model_path, custom_objects={'rmse': rmse})
                    latency, pdr = get_predictions(model_f, rat, gps_data)
                    super_df[f"pred_latency_ms_{rat}_{mt}"] = latency
                    super_df[f"pred_pdr_{rat}_{mt}"] = pdr
                    print(f"  {rat}: {mt} predictions done.")

            print("_ Predictions added.")
            print("_ Sending to the selection algorithm.")
            for mt in model_types:
                super_df[f"Best_RAT_{mt}"] = super_df.apply(select_best_rat, args=(mt,), axis=1)
            print("_ Adding opportunistic algorithm.")
            super_df = opportunistic_best_rat(super_df)
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
        make_rmse_plot()
        make_rmse_table()
        make_table(INPUT)

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
