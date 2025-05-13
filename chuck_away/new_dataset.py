import argparse
import pandas as pd
import numpy as np
import os


def chunk_time_series(df, sequence_length=1536, stride=1536):
    temps = df['temperature'].values
    sequences = []

    for i in range(0, len(temps) - sequence_length + 1, stride):
        seq = temps[i:i + sequence_length]
        sequences.append(seq)

    sequences = np.stack(sequences)
    return sequences


def main(input_path, output_path, sequence_length=1536, stride=1536):
    # Load and sort the data
    df = pd.read_csv(input_path, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Slice into sequences
    sequences = chunk_time_series(df, sequence_length=sequence_length, stride=stride)

    # Save as CSV
    output_df = pd.DataFrame(sequences)
    output_df.to_csv(output_path, index=False)
    print(f"Saved {len(sequences)} sequences of length {sequence_length} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk time series data into sequences for VAE training.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file (with 'time' and 'temperature')")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file (chunked sequences)")
    parser.add_argument("--sequence_length", type=int, default=1536, help="Length of each time series sequence")
    parser.add_argument("--stride", type=int, default=1536, help="Stride between sequences")

    args = parser.parse_args()

    main(args.input, args.output, args.sequence_length, args.stride)
