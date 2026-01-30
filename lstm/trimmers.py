import re
import sys
import argparse

from config import MIN_LAT, MIN_LON

def trim_pc5(input_file, output_file):
    out = 0
    pc5_data={}
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header
        outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,latency_ms\n")

        for line in infile:
            if not line[0].isdigit():
                continue  # Ignore non-digit starting lines

            parts = line.strip().split(',')
            latency = parts[10].strip()
            if latency.startswith("0"):
                continue  # Skip malformed lines

            tx_seq_num = str(int(parts[15].strip()))
            tx_timestamp = str(float(parts[17].strip())/1000.0) # convert to ms
            tx_latitude = parts[20].strip()
            tx_longitude = parts[21].strip()

            pc5_data[int(float(tx_timestamp))//1000] = (tx_latitude, tx_longitude)

            outfile.write(f"{tx_seq_num},{tx_timestamp},{tx_latitude},{tx_longitude},{latency}\n")
            out += 1

    return out, pc5_data

def trim_dsrc(input_file, output_file, pc5_data=None):
    out = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Write header
        if pc5_data is None:
            outfile.write("tx_seq_num,tx_timestamp_ms,rsrp_1,rsrp_2,latency_ms\n")
        else:
            outfile.write("tx_seq_num,tx_timestamp_ms,tx_latitude,tx_longitude,rsrp_1,rsrp_2,latency_ms\n")

        for line in infile:
            parts = re.split(r'\s+', line.strip())
            #if len(parts) != 17: #TODO uncomment after testing
            if len(parts) < 16 or len(parts) > 17:
                continue  # Skip malformed lines and headers
            #elif len(parts) == 16: #TODO uncomment after testing
            #    print ("This DSRC logfile does not contain latency values. Are you sure you enabled the -g time flag when running llc test-tx?")
            #    raise ValueError("Invalid log format, missing latency values. Output file will be broken.")

            tx_seq_num = str(int(parts[0])) # get rid of leading zeros
            timestamp = str(float(parts[9])*1000.0) # convert to ms
            power_ant1 = parts[5]
            power_ant2 = parts[6]
            latency = parts[10]

            if power_ant1.startswith("1"): # circumvent a bug where 16383.0 value is logged instead of -102.0 dBm
                power_ant1="-102.0"

            if pc5_data is None:
                outfile.write(f"{tx_seq_num},{timestamp},{power_ant1},{power_ant2},{latency}\n")

            else:
                ts = int(float(timestamp))//1000 # we transmitted every 100ms, we round up for every second
                if ts not in pc5_data: # floor to nearest timestamp in ms
                    if ts+1 in pc5_data:
                        latitude, longitude = pc5_data.get(ts+1, ("", ""))
                    elif ts-1 in pc5_data:
                        latitude, longitude = pc5_data.get(ts-1, ("", ""))
                    else:
                        latitude, longitude = MIN_LAT, MIN_LON
                else:
                    latitude, longitude = pc5_data.get(ts, ("", ""))
                outfile.write(f"{tx_seq_num},{timestamp},{latitude},{longitude},{power_ant1},{power_ant2},{latency}\n")

            out += 1

    return out




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim log files for PC5 and DSRC",
        epilog="Examples:\n"
               "  python trimmers.py --rat pc5 --input_file input_pc5.log --output_file output_pc5.log\n"
               "  python trimmers.py --rat dsrc --input_file input_dsrc.log --output_file output_dsrc.log\n"
               "  python trimmers.py --rat both --input_file input_dsrc.log --output_file output_dsrc.log --input_file_both input_pc5.log --output_file_both output_pc5.log",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--rat', type=str, required=True, choices=['dsrc', 'pc5', 'both'], help="Type of log file to trim (dsrc, pc5, or both)")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input log file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output trimmed log file")
    parser.add_argument('--input_file_both', type=str, help="Path to the input PC5 log file (required if type is 'both')")
    parser.add_argument('--output_file_both', type=str, help="Path to the output trimmed PC5 log file (required if type is 'both')")
    args = parser.parse_args()

    if args.rat == "dsrc":
        out = trim_dsrc(args.input_file, args.output_file)
        print("Generated " + str(out) + " lines")
    elif args.rat == "pc5":
        out,data = trim_pc5(args.input_file, args.output_file)
        print("Generated " + str(out) + " lines")
    elif args.rat == "both":
        if not args.input_file_both or not args.output_file_both:
            print("Both input_file_both and output_file_both are required when rat is 'both'")
            sys.exit(1)
        out_pc5, pc5_data = trim_pc5(args.input_file_both, args.output_file_both)
        out_dsrc = trim_dsrc(args.input_file, args.output_file, pc5_data)
        print("Generated " + str(out_pc5) + " lines for PC5")
        print("Generated " + str(out_dsrc) + " lines for DSRC")
    else:
        print("Invalid rat type, must be 'dsrc', 'pc5', or 'both'.")
        sys.exit(1)