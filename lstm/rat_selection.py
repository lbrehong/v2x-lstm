import os.path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from tabulate import tabulate
from main import get_latest_model
from model import rmse, automatic_train
from data_preprocessing import preprocess_lstm_input
import folium
from folium.plugins import PolyLineTextPath
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import argparse

#%%
MODEL_DIR = "models"
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176
MIN_LATENCY, MAX_LATENCY = 4,50 # in ms
TIMESTEPS = 10
TARGET_COLS = ['latency_ms', 'pdr']
DIR = "/home/ray/Documents/icccn-lstm/logs_cohda/obs/"
RATS = {"dsrc": "DSRC", "pc5": "C-V2X PC5", "5g": "5G SA"}
MODELS = ["lstm", "gru", "rnn"]
#RATS = {"dsrc": "DSRC"}
#MODELS = ["lstm"]

gps_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]])
latency_scaler = MinMaxScaler(feature_range=(0, 1)).fit([[MIN_LATENCY], [MAX_LATENCY]])


# Load and preprocess CSV
def grab_gps(df):
    latitude = df["tx_latitude"]
    longitude = df["tx_longitude"]
    dfr = pd.DataFrame()
    dfr["tx_latitude"] = latitude
    dfr["tx_longitude"] = longitude
    dfr = dfr.drop_duplicates()
    print(f"Loaded {len(dfr)} GPS points")
    return dfr


# Predict using each model
def get_predictions(model, rat, gps_data):
    print(f"Prediction over {len(gps_data)} GPS points")

    # Normalize GPS data
    gps_data_scaled = gps_scaler.transform(gps_data)
    sequence_length = 10  # Model expects 10 time steps
    if rat != "pc5":
        num_features = 6  # Model expects 6 features per time step
    else:
        num_features = 4

    # Placeholder for input sequences
    input_sequences = []

    # Construct rolling sequences
    for i in range(len(gps_data_scaled)):
        past_points = gps_data_scaled[max(0, i-sequence_length+1):i+1]
        past_points_full = np.hstack((past_points, np.zeros((past_points.shape[0], num_features - past_points.shape[1]))))

        if len(past_points) < sequence_length:
            padding = np.zeros((sequence_length - len(past_points), num_features))
            sequence = np.vstack((padding, past_points_full))
        else:
            sequence = past_points_full

        input_sequences.append(sequence)

    # Convert list to numpy array
    input_sequences = np.array(input_sequences)

    # Debug: Check input shape before model prediction
    print("Input Sequences Shape:", input_sequences.shape)
    if input_sequences.shape[0] == 0:
        raise ValueError("Error: No input sequences generated. Check GPS preprocessing.")

    # Predict using the LSTM model
    predictions = model.predict(input_sequences)

    # Convert predictions to NumPy array
    predictions = np.array(predictions)

    # Ensure the correct shape
    #print("Raw Predictions Shape:", predictions.shape)  # Debugging

    # If the output shape is (13383, 10, 2), select the last prediction of each sequence
    if predictions.shape == (2, len(gps_data), 1):  # If model gives multiple timesteps per input
        predictions = predictions.reshape(len(gps_data), 2) # Take the last timestep's output

    print("Processed Predictions Shape:", predictions.shape)
    #print("First few processed predictions:\n", predictions[:5])

    # Extract latency and PDR
    pred_latency, pred_pdr = predictions[:, 0].flatten(), predictions[:, 1].flatten()  # Extract first values from both arrays
    latency = latency_scaler.inverse_transform(np.array(pred_latency).reshape(-1, 1)).flatten()
    pdr = np.clip(pred_pdr, 0.0, 1.0)  # Cap PDR at 1.0

    print("First few latency predictions:", latency[:5])
    print("First few PDR predictions:", pdr[:5])

    return latency, pdr

def merge_csvs(directory=DIR):
    # Load the base super-CSV (5G data) and the actual matched PC5/DSRC CSVs
    df_super = pd.read_csv(os.path.join(directory,"super.csv"))
    df_base_dsrc = pd.read_csv(os.path.join(directory,"matched_dsrc.csv"), usecols=["tx_latitude", "tx_longitude", "latency_ms", "pdr"])
    df_base_pc5 = pd.read_csv(os.path.join(directory,"matched_pc5.csv"), usecols=["tx_latitude", "tx_longitude", "latency_ms", "pdr"])
    df_base_dsrc.rename(columns={"latency_ms": "latency_ms_dsrc", "pdr": "pdr_dsrc"}, inplace=True)
    df_base_pc5.rename(columns={"latency_ms": "latency_ms_pc5", "pdr": "pdr_pc5"}, inplace=True)

    # Merge the base DSRC and PC5 data into the 5G dataframe
    df_super = df_super.merge(df_base_dsrc, on=["tx_latitude", "tx_longitude"], how="left")
    df_super = df_super.merge(df_base_pc5, on=["tx_latitude", "tx_longitude"], how="left")

    df_super.to_csv(os.path.join(directory,"super_merged.csv"))
    return df_super


def add_predictions(df,directory=DIR):
    # Load DSRC, PC5, 5G predictions
    for model in MODELS:
        for rat in RATS:
            filename = os.path.join(directory,"final_log_" + model + "_" + rat + ".csv")

            try:
                df_pred = pd.read_csv(filename, usecols=["latitude", "longitude", "pred_latency", "pred_pdr"]) # TODO filename
            except FileNotFoundError:
                df_pred = pd.read_csv("final_log_" + model + "_" + rat + ".csv", usecols=["latitude", "longitude", "pred_latency", "pred_pdr"]) # TODO filename
            df_pred.rename(columns={"latitude": "tx_latitude", "longitude": "tx_longitude", "pred_latency": f"pred_latency_ms_{rat}_{model}", "pred_pdr": f"pred_pdr_{rat}_{model}"}, inplace=True)

            # Merge DSRC and PC5 data into the 5G dataframe
            df = df.merge(df_pred, on=["tx_latitude", "tx_longitude"], how="left")
    return df


# Decision algorithm
def select_best_rat(row,model):
    options = [
        ("dsrc", row[f"pred_latency_ms_dsrc_{model}"], row[f"pred_pdr_dsrc_{model}"], row["pdr_dsrc"]),
        ("pc5", row[f"pred_latency_ms_pc5_{model}"], row[f"pred_pdr_pc5_{model}"], row["pdr_pc5"]),
        ("5g", row[f"pred_latency_ms_5g_{model}"], row[f"pred_pdr_5g_{model}"], row["pdr_5g"]),
    ]

    # Filter by PDR threshold
    valid_options = [opt for opt in options if opt[2] >= 0.99]
    if not valid_options:
        keep = [opt for opt in options if opt[3] >= 0.1] # Filter out the unavailable RATs
        if keep:
            best_rat = max(keep, key=lambda x: x[2])[0]  # Select RAT with highest PDR
        elif options[2][3] >= 0.1:
            best_rat = "5g"
        else:
            best_rat = "NaN"

    else:
        # Select lowest latency
        valid_options.sort(key=lambda x: x[1])
        best_rat = valid_options[0][0]

        # Handle latency tie-breaking
        if len(valid_options) > 1:
            if abs(valid_options[0][1] - valid_options[1][1]) < 1:
                if "5g" in [valid_options[0][0], valid_options[1][0]]:
                    best_rat = "5g"
                elif "pc5" in [valid_options[0][0], valid_options[1][0]]:
                    best_rat = "pc5"
    #print(f"Row {row.name}: Selected {best}")
    return best_rat

def opportunistic_best_rat(df):
    best_rat_list = []
    previous_rat = "5g"
    for index, row in df.iterrows():
        options = [
            ("dsrc", row["latency_ms_dsrc"], row["pdr_dsrc"]),
            ("pc5", row["latency_ms_pc5"], row["pdr_pc5"]),
            ("5g", row["latency_ms_5g"], row["pdr_5g"]),
        ]
        # Filter by PDR threshold
        valid_options = [opt for opt in options if opt[2] > 0.05] # Filter out <20% PDR
        if not valid_options:
            keep = [opt for opt in options if opt[2] > 0.0] # Filter out the unavailable RATs
            if keep:
                best_rat = min(keep, key=lambda x: x[1])[0]  # Select RAT with lowest latency
                best_rat_list.append(best_rat)
                previous_rat = best_rat  # Update last selected RAT
                continue
            elif options[2][2] > 0.1:
                best_rat = "5g"
                best_rat_list.append(best_rat)
                previous_rat = best_rat  # Update last selected RAT
                continue
            else:
                best_rat = "NaN" # TODO default
                best_rat_list.append(best_rat)
                previous_rat = best_rat  # Update last selected RAT
                continue

        # Step 2: If only 5G is available, select it
        if len(valid_options) == 1 and valid_options[0][0] == "5g":
            best_rat = "5g"
        # Step 3: If DSRC and/or PC5 is available and previous RAT was 5G, pick the lowest latency
        elif previous_rat == "5g" and any(rat[0] in ["dsrc", "pc5"] for rat in valid_options):
            best_rat = min(valid_options, key=lambda x: x[1])[0]  # Choose lowest latency
        # Step 4: If previous RAT is available again, keep the previous choice
        elif any(rat[0] == previous_rat for rat in valid_options):
            best_rat = previous_rat
        elif options[2][2] > 0.1:
            best_rat = "5g"
        else:
            best_rat = "NaN" # TODO default

        best_rat_list.append(best_rat)
        previous_rat = best_rat  # Update last selected RAT

    df["Best_RAT_opp"] = best_rat_list
    return df


# Main processing function
def process_all(input_dir=DIR):
    for rat in RATS:
        dfl = pd.read_csv(os.path.join(input_dir, "matched_" + rat + ".csv"))

        print("___ Starting data preprocessing.")
        print("______ Final set.")
        (X_new,y_new,scalers) = preprocess_lstm_input(dfl, new=True, rat=rat, target_cols=TARGET_COLS, seq_length=TIMESTEPS)
        print("___ Preprocessing complete.")

        for model in MODELS:
            # Load trained LSTM models
            model_f = load_model(get_latest_model(model, rat), custom_objects={'rmse': rmse})

            print("____________________________________________________")
            print("Automatic retraining.")
            automatic_train(model_f, X_new, y_new, 32, 200, 0.15, "final_log_" + model + "_" + rat + ".csv", rat, model)

            print(f"{rat}: {model} predictions done.")



# Main processing function
def process_file(input_csv, model_type, output_csv):
    dfl = pd.read_csv(os.path.join(DIR,input_csv))
    df_gps = grab_gps(dfl)
    df = pd.DataFrame()
    df["tx_latitude"] = df_gps["tx_latitude"]
    df["tx_longitude"] = df_gps["tx_longitude"]

    print("___ Starting data preprocessing.")
    print("______ Training set.")
    (X_5g,y_5g,scalers) = preprocess_lstm_input(df, new=True, rat="5g", target_cols=TARGET_COLS, seq_length=TIMESTEPS)
    (X_pc5,y_pc5,scalers) = preprocess_lstm_input(df, new=True, rat="pc5", target_cols=TARGET_COLS, seq_length=TIMESTEPS)
    (X_dsrc,y_dsrc,scalers) = preprocess_lstm_input(df, new=True, rat="dsrc", target_cols=TARGET_COLS, seq_length=TIMESTEPS)

    # Load trained LSTM models
    model_dsrc = load_model(get_latest_model(model_type, "dsrc"), custom_objects={'rmse': rmse})
    model_cv2x = load_model(get_latest_model(model_type, "pc5"), custom_objects={'rmse': rmse})
    model_5g = load_model(get_latest_model(model_type, "5g"), custom_objects={'rmse': rmse})

    print("GPS Data shape:", df.shape)
    print("First few GPS entries:\n", df[:5])

    # Get predictions
    df_pred = pd.DataFrame()
    df_pred["latency_DSRC"], df_pred["PDR_DSRC"] = get_predictions(model_dsrc, "dsrc", df)
    df_pred["latency_CV2X"], df_pred["PDR_CV2X"] = get_predictions(model_cv2x, "pc5", df)
    df_pred["latency_5G"], df_pred["PDR_5G"] = get_predictions(model_5g, "5g", df)

    # Select best RAT
    df[f"Best_RAT_{model_type}"] = df_pred.apply(select_best_rat, axis=1)

    # Save results
    output_csv = os.path.join(DIR, "output_" + input_csv)
    df.to_csv(output_csv, index=False)
    print(f"Processed file saved to {output_csv}")


# Visualization function
def visualize_rat_map(output_csv, map_output="rat_map"):

    # Define color mapping for RAT types
    color_map = {"dsrc": "blue", "pc5": "orange", "5g": "green"}
    MODELS.append("opp")

    for model in MODELS:

        # Initialize the map at the first GPS point
        df = pd.read_csv(output_csv)
        start_location = [df.iloc[0]["tx_latitude"], df.iloc[0]["tx_longitude"]]
        m = folium.Map(location=start_location, zoom_start=14, tiles="OpenStreetMap")

        # Draw the path with color-coded segments
        for i in range(len(df) - 1):
            lat1, lon1, rat1 = df.iloc[i][["tx_latitude", "tx_longitude", f"Best_RAT_{model}"]]
            lat2, lon2, rat2 = df.iloc[i + 1][["tx_latitude", "tx_longitude", f"Best_RAT_{model}"]]
            folium.PolyLine([(lat1, lon1), (lat2, lon2)],
                            color=color_map.get(rat1, "gray"),
                            weight=5,
                            opacity=0.8).add_to(m)

        # Save the map to an HTML file
        m.save(os.path.join(DIR, f"{map_output}_{model}.html"))
        print(f"Map saved to {map_output}_{model}.html")


# Function to extract latencies based on the selected RAT for each scheme
def get_latencies(df, scheme_column):
    return df.apply(lambda row: row[f"latency_ms_{row[scheme_column].lower()}"]
                    if pd.notna(row[scheme_column]) else None, axis=1)

def get_pdr(df, scheme_column):
    return df.apply(lambda row: row[f"pdr_{row[scheme_column].lower()}"] * 100
                    if pd.notna(row[scheme_column]) else None, axis=1)


def make_histogram_latency(input_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Define latency bins and labels
    bins = [0, 10, 20, 50, float("inf")]
    labels = ["<10ms", "10-20ms", "20-50ms", ">50ms"]

    # Extract latency values for each scheme
    latency_lstm = get_latencies(df, "Best_RAT_lstm")
    latency_gru = get_latencies(df, "Best_RAT_gru")
    latency_rnn = get_latencies(df, "Best_RAT_rnn")
    latency_opportunistic = get_latencies(df, "Best_RAT_opp")

    # Create a DataFrame to store categorized latencies
    category_data = pd.DataFrame({
        "LSTM": pd.cut(latency_lstm, bins=bins, labels=labels),
        "GRU": pd.cut(latency_gru, bins=bins, labels=labels),
        "RNN": pd.cut(latency_rnn, bins=bins, labels=labels),
        "No pQoS": pd.cut(latency_opportunistic, bins=bins, labels=labels),
    })

    # Compute the percentage distribution for each category
    hist_data = category_data.apply(lambda x: x.value_counts(normalize=True) * 100)

    # Plot histogram
    ax = hist_data.plot(kind="bar", figsize=(10, 6), width=0.8)
    plt.title("Latency Distribution by Scheme")
    plt.xlabel("Latency Category")
    plt.ylabel("Percentage of Total Transmissions")
    plt.xticks(rotation=0)
    plt.legend(title="Selection Scheme")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(DIR, "latency_histogram.png"))


def make_histogram_pdr(input_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Define latency bins and labels
    bins = [0.0, 95.0, 99.0, 99.9, 100.0]
    labels = ["<95%", "99-95%", "99.9-99%", ">99.9%"]

    # Extract latency values for each scheme
    pdr_lstm = get_pdr(df, "Best_RAT_lstm")
    pdr_gru = get_pdr(df, "Best_RAT_gru")
    pdr_rnn = get_pdr(df, "Best_RAT_rnn")
    pdr_opportunistic = get_pdr(df, "Best_RAT_opp")

    # Create a DataFrame to store categorized latencies
    category_data = pd.DataFrame({
        "LSTM": pd.cut(pdr_lstm, bins=bins, labels=labels),
        "GRU": pd.cut(pdr_gru, bins=bins, labels=labels),
        "RNN": pd.cut(pdr_rnn, bins=bins, labels=labels),
        "No pQoS": pd.cut(pdr_opportunistic, bins=bins, labels=labels),
    })

    # Compute the percentage distribution for each category
    hist_data = category_data.apply(lambda x: x.value_counts(normalize=True) * 100).reindex(labels[::-1])
    # Compute raw count values for each bin
    count_data = category_data.apply(lambda x: x.value_counts()).reindex(labels[::-1])
    # Compute confidence intervals
    ci_ranges = count_data.applymap(lambda n: (st.t.interval(0.95, df=n-1, loc=n, scale=np.sqrt(n))[1] - n) if n > 1 else 0)
    # Convert CI values to percentage scale
    ci_ranges = (ci_ranges / count_data.sum()) * 100

    # Plot histogram with error bars (CIs)
    fig, ax = plt.subplots(figsize=(20, 12))
    hist_data.plot(kind="bar", yerr=ci_ranges, capsize=5, ax=ax, width=0.8, error_kw={'elinewidth': 2, 'alpha': 0.6})

    # Plot histogram
    plt.title("PDR Distribution by Model Type", fontsize=28)
    plt.xlabel("PDR Category", fontsize=24)
    plt.ylabel("Percentage of Total Transmissions (%)", fontsize=28)
    plt.xticks(rotation=0, fontsize=22)
    plt.yticks(np.arange(0, 101, 10), fontsize=22)  # Set y-axis increments to 10
    plt.legend(title="Model Type", fontsize=22, title_fontsize=24)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()
    plt.savefig(os.path.join(DIR, "pdr_histogram.png"))



# Function to calculate mean and confidence interval
def mean_ci(series, confidence=0.95):
    series = series.dropna()  # Remove NaN values
    mean = np.mean(series)
    if len(series) > 1:
        ci = st.t.interval(confidence, len(series)-1, loc=mean, scale=st.sem(series))
        ci_range = ci[1] - mean  # Compute the upper range of CI
    else:
        ci_range = 0  # If only one value, CI is undefined
    return mean, ci_range

# Function to extract PDR and latency based on selected RAT
def get_metric(df, scheme_column, metric):
    return df.apply(lambda row: row[f"{metric}_{row[scheme_column].lower()}"]
                    if pd.notna(row[scheme_column]) else None, axis=1)


def make_table(input_csv):
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Calculate metrics for each scheme
    summary_data = {}
    schemes = ["Best_RAT_lstm", "Best_RAT_gru", "Best_RAT_rnn", "Best_RAT_opp"]

    for scheme in schemes:
        scheme_name = scheme.replace("Best_RAT_", "").upper()

        # Extract PDR and latencies based on the selected RAT for each scheme
        pdr_values = get_pdr(df, scheme).copy()
        latency_values = get_latencies(df, scheme).copy()

        # Compute statistics for PDR
        avg_pdr, ci_pdr = mean_ci(pdr_values)
        avg_latency, ci_latency = mean_ci(latency_values)
        max_latency = np.max(latency_values.dropna()) if not latency_values.dropna().empty else np.nan

        # Compute **average latencies per RAT within this scheme only**
        rat_latencies = {
            "dsrc": get_latencies(df[df[scheme] == "dsrc"], scheme),
            "pc5": get_latencies(df[df[scheme] == "pc5"], scheme),
            "5g": get_latencies(df[df[scheme] == "5g"], scheme),
        }

        avg_dsrc_latency, ci_dsrc = mean_ci(rat_latencies["dsrc"])
        avg_pc5_latency, ci_pc5 = mean_ci(rat_latencies["pc5"])
        avg_5g_latency, ci_5g = mean_ci(rat_latencies["5g"])

        # Compute the percentage of messages sent over each RAT
        total_messages = len(df)
        rat_counts = df[scheme].value_counts(normalize=True) * 100  # Normalize to get percentage

        pct_dsrc = (df[scheme] == "dsrc").sum() / total_messages * 100
        pct_pc5 = (df[scheme] == "pc5").sum() / total_messages * 100
        pct_5g = (df[scheme] == "5g").sum() / total_messages * 100

        # Store in dictionary
        summary_data[scheme_name] = [
            (avg_pdr, ci_pdr),
            (avg_latency, ci_latency),
            (avg_dsrc_latency, ci_dsrc),
            (avg_pc5_latency, ci_pc5),
            (avg_5g_latency, ci_5g),
            (max_latency, 0),  # No CI needed for max value
            (pct_dsrc, 0),  # DSRC percentage
            (pct_pc5, 0),  # PC5 percentage
            (pct_5g, 0)  # 5G percentage
        ]

    # Convert to DataFrame for presentation
    summary_df = pd.DataFrame(summary_data, index=[
        "Avg PDR", "Avg Latency", "Avg DSRC Latency", "Avg PC5 Latency", "Avg 5G Latency", "Max Latency", "DSRC Usage (%)", "PC5 Usage (%)", "5G Usage (%)"
    ])

    # Format CI values nicely
    summary_df = summary_df.applymap(lambda x: f"{x[0]:.3f} ± {x[1]:.3f}" if isinstance(x, tuple) else x)

    # Display the summary table
    print(tabulate(summary_df, headers="keys", tablefmt="pretty"))
    summary_df.to_csv(os.path.join(os.path.dirname(input_csv),"best_perf_table.csv"), index=False)


def make_rmse_table(input_csv="/home/ray/Documents/icccn-lstm/devs/lstm/"):
    schemes = ["Best_RAT_lstm", "Best_RAT_gru", "Best_RAT_rnn"]
    # Initialize dictionary to store results
    rmse_summary = {}

    # Compute average RMSE (Latency & PDR) per model and RAT
    for model in MODELS:
        scheme_data = []
        for rat in RATS:

            # Load the CSV file
            df = pd.read_csv(os.path.join(input_csv, "final_log_" + model + "_" + rat + ".csv"))
            # Compute statistics for Latency RMSE
            latency_rmse, ci_latency = mean_ci(df[f"rmse_latency"])
            # Compute statistics for PDR RMSE
            pdr_rmse, ci_pdr = mean_ci(df[f"rmse_pdr"])

            scheme_data.append((f"{latency_rmse:.3f} ± {ci_latency:.3f}", f"{pdr_rmse:.3f} ± {ci_pdr:.3f}"))

        rmse_summary[model] = scheme_data

    # Convert to DataFrame for better presentation
    rmse_summary_df = pd.DataFrame(rmse_summary, index=["DSRC", "PC5", "5G"])

    # Set proper column labels
    rmse_summary_df.columns = ["LSTM", "GRU", "RNN"]
    rmse_summary_df.index.name = "RAT"

    # Format output for better readability
    #rmse_summary_df.columns = pd.MultiIndex.from_product([["Latency RMSE", "PDR RMSE"], rmse_summary_df.columns])
    rmse_summary_df.to_csv(os.path.join(os.path.dirname(input_csv),"best_rmse_table.csv"))

def make_rmse_plot(input_csv="/home/ray/Documents/icccn-lstm/devs/lstm/"):
    # Define the moving average window
    window_size = 250  # Adjust as needed

    # Define schemes and corresponding subplot indices
    schemes = ["Best_RAT_lstm", "Best_RAT_gru", "Best_RAT_rnn"]
    titles = ["LSTM", "GRU", "SimpleRNN"]
    colors = {"dsrc": "blue", "pc5": "orange", "5g": "green"}

    # Create 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

    # Iterate over each scheme and plot RMSE trends
    i=0
    for model in MODELS:
        ax_lat = axes[0,i]
        ax_pdr = axes[1,i]
        for rat in RATS:

            # Load the CSV file
            df = pd.read_csv(os.path.join(input_csv, "final_log_" + model + "_" + rat + ".csv"))


            # Compute moving average for Latency RMSE
            latency_rmse_ma = df[f"rmse_latency"].rolling(window=window_size, min_periods=1).mean()
            ax_lat.plot(latency_rmse_ma, label=f"Latency RMSE {rat.upper()}", linestyle="-", color=colors[rat])

            # Compute moving average for PDR RMSE
            pdr_rmse_ma = df[f"rmse_pdr"].rolling(window=window_size, min_periods=1).mean()
            ax_pdr.plot(pdr_rmse_ma, label=f"PDR RMSE {rat.upper()}", linestyle="-", color=colors[rat])

        # Customize latency RMSE subplot
        ax_lat.set_title(titles[i])
        ax_lat.set_xlabel("Message Index")
        ax_lat.set_ylabel("Latency RMSE")
        ax_lat.set_ylim(0, 15)
        ax_lat.grid(True, linestyle="--", alpha=0.5)
        ax_lat.legend()

        # Customize PDR RMSE subplot
        ax_pdr.set_title(titles[i])
        ax_pdr.set_xlabel("Message Index")
        ax_pdr.set_ylabel("PDR RMSE")
        ax_pdr.set_ylim(0, 1)
        ax_pdr.grid(True, linestyle="--", alpha=0.5)
        ax_pdr.legend()

        i+=1
    # Adjust layout and show plot
    plt.tight_layout()
    plt.suptitle("Latency & PDR RMSE Moving Averages per Scheme", fontsize=14, y=1.05)
    plt.show()
    plt.savefig(os.path.join(input_csv, "rmse_pred_plot.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select best RAT for given GPS coordinates")
    parser.add_argument('--input', type=str, required=True, help="Path to the input CSV folder")
    parser.add_argument('--mode', type=str, help="Is it time to calculate (empty), to check GPS data (test), or to visualize (view)?")
    parser.add_argument('--model_type', type=str, help="RNN model type")
    args = parser.parse_args()

    INPUT = args.input
    MODEL_TYPE = args.model_type
    MODE = args.mode

    if not MODE and MODEL_TYPE:
        if MODEL_TYPE != "all":
            process_file(INPUT, MODEL_TYPE, os.path.join(os.path.dirname(INPUT), "selection_results.csv"))
        else:
            print("__________________________________")
            print("____________ LET'S GO ____________")
            print("__________________________________")
            print("_ Processing all input CSVs and getting predictions from each model")
            process_all(INPUT)
            print("_ Processing done.")
            print("_ Merging into the super-CSV.")
            super_df = merge_csvs()
            print("_ Adding predictions into the super-CSV.")
            super_pred_df = add_predictions(super_df)
            print("_ Merging done.")
            print("_ Sending to the selection algorithm.")
            super_df["Best_RAT_lstm"] = super_pred_df.apply(select_best_rat, args=("lstm",), axis=1)
            super_df["Best_RAT_gru"] = super_pred_df.apply(select_best_rat, args=("gru",), axis=1)
            super_df["Best_RAT_rnn"] = super_pred_df.apply(select_best_rat, args=("rnn",), axis=1)
            print("_ Adding opportunistic algorithm.")
            super_df = opportunistic_best_rat(super_df)
            print("_ Algorithms done.")
            print("_ Saving. " + os.path.join(INPUT,"bestRAT_super.csv"))
            super_df.to_csv(os.path.join(INPUT,"bestRAT_super.csv"), index=False)
            print("_ All done.")


    elif MODE == "view":
        visualize_rat_map(INPUT)

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

        # Compute map center
        map_center = [latitude.mean(), longitude.mean()]
        print(map_center)

        # Create a folium map with satellite tiles
        m = folium.Map(
            location=map_center,
            zoom_start=14,
            tiles="Esri.WorldImagery",  # High-quality satellite view
            attr="Esri"
        )

        # Add GPS points to the map
        for lat, lon in zip(latitude, longitude):
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,  # Adjust marker size
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
            ).add_to(m)

        # Save map to an HTML file and display it
        m.save("gps_map_pc5_01.html")

    elif MODE:
        raise ValueError("Invalid mode specified. test, view and <empty> are valid options.")

    else:
        raise ValueError("No model type specified. all, lstm, gru, rnn are valid options.")


#%%