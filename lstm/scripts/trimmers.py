"""
Raw V2X and 5G log file trimming and preprocessing.

This module processes raw log files from vehicular network experiments,
extracting relevant fields and performing necessary corrections:

- 5G SA: Clock drift compensation via linear regression, SINR/RSRP extraction
- PC5/C-V2X: Latency and GPS extraction from sidelink logs
- DSRC: Power level extraction with GPS matching from PC5/5G data

The output files are standardized CSVs ready for the prediction pipeline.

Usage:
    python -m scripts.trimmers --folder /path/to/raw_logs
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import argparse

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

from config import MIN_LAT, MIN_LON
from utils import find_files_with_string

# Drift compensation parameters (calibrated per dataset)
# XMIN, XMAX: latency range for scaling; TH: threshold for drift calculation
XMIN, XMAX, TH = 1.862, 1.921, -1.6  # Calibrated for Dataset 05


def calculate_coefficient(latencies, tx_seq_nums):
    """
    Calculate clock drift coefficient using linear regression.

    Clock drift occurs when transmitter and receiver clocks are not
    synchronized, causing measured latency to increase/decrease linearly
    over time.

    Args:
        latencies: Array of measured latency values (filtered for outliers)
        tx_seq_nums: Corresponding sequence numbers

    Returns:
        Drift coefficient (slope) in ms per packet
    """
    X = np.array(tx_seq_nums).reshape(-1, 1)
    y = np.array(latencies)
    model = LinearRegression()
    model.fit(X, y)
    coefficient = model.coef_[0]
    print(f"Clock drift coefficient: {coefficient:.6f} ms/packet")
    return coefficient


def compensate_drift(data, seqnum, drift=0.0):
    """
    Apply drift compensation to raw latency measurement.

    Args:
        data: Raw latency value (negative due to measurement convention)
        seqnum: Packet sequence number
        drift: Drift coefficient from calculate_coefficient()

    Returns:
        Drift-compensated latency value
    """
    return -(data + seqnum * abs(drift))


def scale_values(data, xmin=XMIN, xmax=XMAX, a=16.0, b=45.0):
    """
    Scale latency values from measured range to expected range.

    Uses min-max normalization to transform values from [xmin, xmax]
    to [a, b] range based on expected latency bounds.

    Args:
        data: Input value to scale
        xmin: Minimum of source range (calibrated per dataset)
        xmax: Maximum of source range (calibrated per dataset)
        a: Minimum of target range (default: 16ms)
        b: Maximum of target range (default: 45ms)

    Returns:
        Scaled latency value in target range
    """
    return a + (data - xmin) * (b - a) / (xmax - xmin)


def trim_sa(file_list, output_file, path):
    """
    Process 5G Standalone (SA) log files with drift compensation.

    Performs:
        1. Two-pass processing: first pass calculates drift coefficient
        2. Applies drift compensation and scaling to latency
        3. Extracts SINR and RSRP signal quality metrics
        4. Builds GPS lookup table for DSRC matching

    Args:
        file_list: List of 5G log filenames to process
        output_file: Path for output trimmed CSV
        path: Directory containing input files

    Returns:
        Dictionary mapping timestamps to (latitude, longitude) tuples
    """
    input_file = os.path.join(path, file_list[0])
    out = 0
    sa_data = {}
    latencies = []
    tx_seq_nums = []
    saved_sinr = 0
    saved_rsrp = 0
    threshold = TH

    with open(input_file, 'r') as infile:
        for line in infile:
            if not line[0].isdigit():
                continue
            parts = line.strip().split(',')
            tx_seq_num = str(int(parts[0].strip()))
            latency = float(parts[4].strip())
            if latency <= threshold:
                latencies.append(latency)
                tx_seq_nums.append(int(tx_seq_num))

    if latencies and tx_seq_nums:
        coefficient = calculate_coefficient(latencies, tx_seq_nums)
        if not coefficient:
            raise ValueError("Invalid coefficient value. Output file will be broken.")
    else:
        raise ValueError("Invalid coefficient value. Output file will be broken.")

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,latency_ms,sinr,rsrp\n")

        for line in infile:
            if not line[0].isdigit():
                continue

            parts = line.strip().split(',')
            tx_seq_num = str(int(parts[0].strip()))
            timestamp = str(float(parts[1].strip()) * 1000.0)
            latitude = parts[2].strip()
            longitude = parts[3].strip()
            latency_raw = parts[4].strip()
            latency = scale_values(compensate_drift(float(latency_raw), int(tx_seq_num), coefficient))

            if parts[5].strip().startswith("N"):
                sinr = saved_sinr
                rsrp = saved_rsrp
            else:
                sinr = str(abs(int(parts[5].strip())))
                rsrp = parts[6].strip()
                saved_sinr = sinr
                saved_rsrp = rsrp

            sa_data[int(float(timestamp)) * 1000] = (latitude, longitude)

            outfile.write(f"{tx_seq_num},{timestamp},{latitude},{longitude},{latency},{sinr},{rsrp},0\n")
            out += 1

    return sa_data


def trim_pc5(input_file, output_file):
    """
    Process PC5/C-V2X sidelink log files.

    Extracts latency and GPS coordinates from PC5 mode 4 logs.
    Filters out invalid entries (zero latency, incomplete fields).

    Args:
        input_file: Path to raw PC5 log file
        output_file: Path for output trimmed CSV

    Returns:
        Tuple of (line_count, gps_dict) where gps_dict maps
        timestamps to (latitude, longitude) tuples
    """
    out = 0
    pc5_data = {}

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line[0].isdigit():
                continue

            parts = line.strip().split(',')

            if len(parts) < 21:
                continue

            latency = parts[10].strip()
            if latency.startswith("0"):
                continue

            tx_seq_num = str(int(parts[15].strip()))
            tx_timestamp = str(float(parts[17].strip()) / 1000.0)
            tx_latitude = parts[20].strip()
            tx_longitude = parts[21].strip()

            pc5_data[int(float(tx_timestamp)) // 1000] = (tx_latitude, tx_longitude)

            outfile.write(f"{tx_seq_num},{tx_timestamp},{tx_latitude},{tx_longitude},{latency}\n")
            out += 1

    return out, pc5_data


def trim_dsrc(input_file, output_file, pc5_data, sa_data):
    """
    Process DSRC (802.11p) log files with GPS matching.

    DSRC logs do not contain GPS coordinates directly. This function
    matches timestamps with PC5 or 5G data to obtain location.

    Args:
        input_file: Path to raw DSRC log file
        output_file: Path for output trimmed CSV
        pc5_data: GPS lookup dictionary from PC5 processing
        sa_data: GPS lookup dictionary from 5G processing

    Returns:
        Number of lines written to output file
    """
    out = 0

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 16 or len(parts) > 17:
                continue

            tx_seq_num = str(int(parts[0]))
            timestamp = str(float(parts[9]) * 1000.0)
            power_ant1 = parts[5]
            power_ant2 = parts[6]
            latency = parts[10]

            if power_ant1.startswith("1"):
                power_ant1 = "-102.0"

            ts = int(float(timestamp)) // 1000
            if ts not in pc5_data:
                if ts + 1 in pc5_data:
                    latitude, longitude = pc5_data.get(ts + 1, ("", ""))
                elif ts - 1 in pc5_data:
                    latitude, longitude = pc5_data.get(ts - 1, ("", ""))
                else:
                    if ts in sa_data:
                        latitude, longitude = sa_data.get(ts, ("", ""))
                    elif ts + 1 in sa_data:
                        latitude, longitude = sa_data.get(ts + 1, ("", ""))
                    elif ts - 1 in sa_data:
                        latitude, longitude = sa_data.get(ts - 1, ("", ""))
                    else:
                        latitude, longitude = MIN_LAT, MIN_LON
            else:
                latitude, longitude = pc5_data.get(ts, ("", ""))

            outfile.write(f"{tx_seq_num},{timestamp},{latitude},{longitude},{power_ant1},{power_ant2},{latency}\n")
            out += 1

    return out


def get_first_timestamp(file):
    """
    Extract the first timestamp from a trimmed file for chronological sorting.

    Args:
        file: Path to trimmed CSV file

    Returns:
        First timestamp as float, or infinity if not found
    """
    with open(file, 'r') as f:
        for line in f:
            if line[0].isdigit():
                return float(line.split(',')[1])
    return float('inf')


def append_files(output_file, files):
    """
    Combine multiple trimmed files into a single CSV with header.

    Args:
        output_file: Path for combined output file
        files: List of trimmed file paths to concatenate
    """
    with open(output_file, 'w') as outfile:
        if "dsrc" in output_file:
            outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,rsrp_1,rsrp_2,latency_ms\n")
        elif "pc5" in output_file:
            outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,latency_ms\n")
        for file in files:
            with open(file, 'r') as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim and combine V2X data")
    parser.add_argument('--folder', type=str, required=True, help="Path to the folder containing the log files")
    args = parser.parse_args()

    PATH = args.folder

    files_dsrc = find_files_with_string(PATH, "dsrc")
    files_pc5 = find_files_with_string(PATH, "pc5")
    files_5g = find_files_with_string(PATH, "5g")
    print("Matching DSRC files: ", files_dsrc)
    print("Matching PC5 files: ", files_pc5)
    print("Matching 5G files: ", files_5g)

    sa_data = trim_sa(files_5g, os.path.join(PATH, "trim_5g.csv"), PATH)

    combined_out_pc5 = 0
    combined_out_dsrc = 0
    trimmed_files_pc5 = []
    trimmed_files_dsrc = []

    for i, file in enumerate(files_pc5):
        output_file_pc5 = f"trim_pc5_{i}.csv"
        output_file_dsrc = f"trim_dsrc_{i}.csv"
        print("Trimming PC5 file ", file)
        out_pc5, pc5_data = trim_pc5(os.path.join(PATH, file), os.path.join(PATH, output_file_pc5))
        print(f"Generated {out_pc5} lines for PC5")
        combined_out_pc5 += out_pc5

        rsu_id = file.split("_")[2]
        try:
            log_dsrc = [f for f in files_dsrc if rsu_id in f][0]
        except IndexError:
            print("No matching DSRC file found for ", file)
            trimmed_files_pc5.append(os.path.join(PATH, output_file_pc5))
            continue

        print("Trimming DSRC file ", log_dsrc)
        out_dsrc = trim_dsrc(os.path.join(PATH, log_dsrc), os.path.join(PATH, output_file_dsrc), pc5_data, sa_data)
        print(f"Generated {out_dsrc} lines for DSRC")
        combined_out_dsrc += out_dsrc
        trimmed_files_pc5.append(os.path.join(PATH, output_file_pc5))
        trimmed_files_dsrc.append(os.path.join(PATH, output_file_dsrc))

    trimmed_files_pc5.sort(key=get_first_timestamp)
    append_files(os.path.join(PATH, "trim_pc5.csv"), trimmed_files_pc5)
    print(f"Combined trimmed PC5 files into combined_trimmed_pc5.csv. Total lines: {combined_out_pc5}")

    trimmed_files_dsrc.sort(key=get_first_timestamp)
    append_files(os.path.join(PATH, "trim_dsrc.csv"), trimmed_files_dsrc)
    print(f"Combined trimmed DSRC files into combined_trimmed_dsrc.csv. Total lines: {combined_out_dsrc}")
