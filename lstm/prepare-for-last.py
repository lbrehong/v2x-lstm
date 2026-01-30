import pandas as pd
import numpy as np
import os
from data_preprocessing import compute_pdr_rolling

DIR = "/home/ray/Documents/icccn-lstm/logs_cohda/obs/"
PRIMARY = "5g.csv"
CSV = ["dsrc.csv", "pc5.csv"]
TOLERANCE = 0.00001
TX_INTERVAL_MS = 20 # in ms, 50 packets per second
PDR_WINDOW = 10 # in seconds

#%%



#%% MAIN

# Load primary CSV
primary_df = pd.read_csv(os.path.join(DIR, PRIMARY))
compute_pdr_rolling(primary_df, "tx_timestamp_ms", PDR_WINDOW, 46)


# Step 1: Remove duplicates while keeping the lowest latency but preserving the original order
primary_df = primary_df[primary_df["latency_ms"] > 15.999]  # Remove outliers
primary_df = primary_df.loc[
    primary_df.groupby(["tx_latitude", "tx_longitude"])["latency_ms"].apply(lambda x: x.sample(n=1).index[0]) # random instead of targeting the lowest latency
].sort_index()  # Restore original order

primary_df.drop("tx_timestamp_ms", axis=1, inplace=True)
primary_df.drop("tx_timestamp_sec", axis=1, inplace=True)
primary_df.drop("tx_seq_num", axis=1, inplace=True)
primary_df.to_csv(os.path.join(DIR, "matched_" + PRIMARY), index=False)

super_csv = pd.DataFrame()
super_csv["tx_latitude"] = primary_df["tx_latitude"]
super_csv["tx_longitude"] = primary_df["tx_longitude"]
super_csv["latency_ms_5g"] = primary_df["latency_ms"]
super_csv["pdr_5g"] = primary_df["pdr"]
super_csv.to_csv(os.path.join(DIR, "super.csv"), index=False)


for csvs in CSV:
    print("___________________")
    print("Processing", csvs)
    # Load secondary CSV
    secondary_df = pd.read_csv(os.path.join(DIR, csvs))
    compute_pdr_rolling(secondary_df, "tx_timestamp_ms", PDR_WINDOW, TX_INTERVAL_MS)
    secondary_df = secondary_df[secondary_df["latency_ms"] < 300.001]

    # Result container
    matched_rows = []
    matched = 0
    # Step 2: Look up in secondary CSVs while keeping order
    for _, row in primary_df.iterrows():
        lat, lon = row["tx_latitude"], row["tx_longitude"]

        # First, try exact match
        match = secondary_df[
            (np.abs(secondary_df["tx_latitude"] - lat) <= TOLERANCE) &
            (np.abs(secondary_df["tx_longitude"] - lon) <= TOLERANCE)
        ]

        # If no match found, reduce precision once and search again
        if match.empty:
            i = 1
            while match.empty and i < 6: # Tolerate up to 5 meters of error
                match = secondary_df[
                    (np.abs(secondary_df["tx_latitude"] - lat) <= TOLERANCE*10*i) &
                    (np.abs(secondary_df["tx_longitude"] - lon) <= TOLERANCE*10*i)
                ]
                i+=1

            if match.empty:
                print("No match found for", lat, lon)
            else:
                matched_data = match.iloc[0].to_dict()
                matched_data["tx_latitude"] = lat  # Keep primary latitude
                matched_data["tx_longitude"] = lon  # Keep primary longitude
                matched_rows.append(matched_data)
                matched += 1

        # Save the first match found (if any)
        else:
            matched_data = match.iloc[0].to_dict()
            matched_data["tx_latitude"] = lat  # Keep primary latitude
            matched_data["tx_longitude"] = lon  # Keep primary longitude
            matched_rows.append(matched_data)
            matched += 1

    print(f"Total {matched} matched rows found.")
    # Preserve original order by keeping indices intact
    matched_df = pd.DataFrame(matched_rows)

    # Remove duplicates while keeping the lowest latency but preserving the original order
    matched_df = matched_df.loc[
        matched_df.groupby(["tx_latitude", "tx_longitude"])["latency_ms"].apply(lambda x: x.sample(n=1).index[0])
    ].sort_index()  # Restore original order
    matched_df.drop("tx_timestamp_ms", axis=1, inplace=True)
    matched_df.drop("tx_timestamp_sec", axis=1, inplace=True)
    matched_df.drop("tx_seq_num", axis=1, inplace=True)

    # Save matched rows to a new CSV
    matched_df.to_csv(os.path.join(DIR, "matched_" + csvs), index=False)

    print(f"Processing complete. Total {matched} matched rows saved to matched_{csvs}.")
