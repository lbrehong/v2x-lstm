"""
Cross-RAT data matching by GPS coordinates.

This module aligns data from different RATs (5G, DSRC, PC5) based on
GPS location, enabling fair comparison of QoS metrics at the same
physical positions.

The matching process:
1. Uses 5G data as the primary reference (highest sampling rate)
2. For each 5G GPS point, finds corresponding DSRC/PC5 measurements
3. Uses progressive tolerance expansion if exact match not found
4. Creates unified dataset with all RATs at matching locations

Usage:
    python prepare-for-last.py --input /path/to/trimmed_data
"""
import pandas as pd
import numpy as np
import os
import argparse

from config import TX_INTERVAL_MS, PDR_WINDOW
from data_preprocessing import compute_pdr_rolling

# Input file configuration
PRIMARY = "5g.csv"  # Reference RAT for GPS coordinates
CSV = ["dsrc.csv", "pc5.csv"]  # Secondary RATs to match

# GPS matching tolerance (approximately 1 meter at mid-latitudes)
TOLERANCE = 0.00001


def match_data(input_dir):
    """
    Match primary 5G data with secondary DSRC/PC5 data by GPS location.

    Creates aligned datasets where each row represents the same physical
    location across all RATs, enabling direct QoS comparison.

    Process:
        1. Load and filter 5G data (remove outliers, compute PDR)
        2. Remove duplicate GPS points (random sample for latency)
        3. For each secondary RAT, find matching points within tolerance
        4. Expand tolerance progressively if no match found

    Args:
        input_dir: Directory containing trimmed CSV files (5g.csv, dsrc.csv, pc5.csv)

    Outputs:
        - matched_5g.csv: Deduplicated 5G data
        - matched_dsrc.csv: DSRC data matched to 5G locations
        - matched_pc5.csv: PC5 data matched to 5G locations
        - super.csv: Combined reference with 5G latency/PDR
    """
    # Load primary CSV
    primary_df = pd.read_csv(os.path.join(input_dir, PRIMARY))
    compute_pdr_rolling(primary_df, "tx_timestamp_ms", PDR_WINDOW, 46)

    # Remove duplicates while keeping the lowest latency but preserving the original order
    primary_df = primary_df[primary_df["latency_ms"] > 15.999]  # Remove outliers
    primary_df = primary_df.loc[
        primary_df.groupby(["tx_latitude", "tx_longitude"])["latency_ms"].apply(
            lambda x: x.sample(n=1).index[0]
        )
    ].sort_index()

    primary_df.drop("tx_timestamp_ms", axis=1, inplace=True)
    primary_df.drop("tx_timestamp_sec", axis=1, inplace=True)
    primary_df.drop("tx_seq_num", axis=1, inplace=True)
    primary_df.to_csv(os.path.join(input_dir, f"matched_{PRIMARY}"), index=False)

    super_csv = pd.DataFrame()
    super_csv["tx_latitude"] = primary_df["tx_latitude"]
    super_csv["tx_longitude"] = primary_df["tx_longitude"]
    super_csv["latency_ms_5g"] = primary_df["latency_ms"]
    super_csv["pdr_5g"] = primary_df["pdr"]
    super_csv.to_csv(os.path.join(input_dir, "super.csv"), index=False)

    for csvs in CSV:
        print("___________________")
        print("Processing", csvs)
        secondary_df = pd.read_csv(os.path.join(input_dir, csvs))
        compute_pdr_rolling(secondary_df, "tx_timestamp_ms", PDR_WINDOW, TX_INTERVAL_MS)
        secondary_df = secondary_df[secondary_df["latency_ms"] < 300.001]

        matched_rows = []
        matched = 0

        for _, row in primary_df.iterrows():
            lat, lon = row["tx_latitude"], row["tx_longitude"]

            match = secondary_df[
                (np.abs(secondary_df["tx_latitude"] - lat) <= TOLERANCE) &
                (np.abs(secondary_df["tx_longitude"] - lon) <= TOLERANCE)
            ]

            if match.empty:
                i = 1
                while match.empty and i < 6:
                    match = secondary_df[
                        (np.abs(secondary_df["tx_latitude"] - lat) <= TOLERANCE * 10 * i) &
                        (np.abs(secondary_df["tx_longitude"] - lon) <= TOLERANCE * 10 * i)
                    ]
                    i += 1

                if match.empty:
                    print("No match found for", lat, lon)
                else:
                    matched_data = match.iloc[0].to_dict()
                    matched_data["tx_latitude"] = lat
                    matched_data["tx_longitude"] = lon
                    matched_rows.append(matched_data)
                    matched += 1
            else:
                matched_data = match.iloc[0].to_dict()
                matched_data["tx_latitude"] = lat
                matched_data["tx_longitude"] = lon
                matched_rows.append(matched_data)
                matched += 1

        print(f"Total {matched} matched rows found.")
        matched_df = pd.DataFrame(matched_rows)

        matched_df = matched_df.loc[
            matched_df.groupby(["tx_latitude", "tx_longitude"])["latency_ms"].apply(
                lambda x: x.sample(n=1).index[0]
            )
        ].sort_index()
        matched_df.drop("tx_timestamp_ms", axis=1, inplace=True)
        matched_df.drop("tx_timestamp_sec", axis=1, inplace=True)
        matched_df.drop("tx_seq_num", axis=1, inplace=True)

        matched_df.to_csv(os.path.join(input_dir, f"matched_{csvs}"), index=False)
        print(f"Processing complete. Total {matched} matched rows saved to matched_{csvs}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match cross-RAT data by GPS coordinates")
    parser.add_argument('--input', type=str, required=True, help="Path to the input directory")
    args = parser.parse_args()

    match_data(args.input)
