#!/usr/bin/env python3

import argparse
import pandas as pd

def add_prefix_by_index(input_csv, output_csv, morph_range=None, env_range=None, dna_range=None):
    df = pd.read_csv(input_csv)
    cols = df.columns.tolist()
    total_cols = len(cols)

    def prefix_columns_by_index(col_range, prefix):
        if col_range:
            start_idx, end_idx = col_range
            if end_idx is None:
                end_idx = total_cols
            for i in range(start_idx, end_idx):
                cols[i] = f"{prefix}{cols[i]}"

    prefix_columns_by_index(morph_range, "morph_")
    prefix_columns_by_index(env_range, "env_")
    prefix_columns_by_index(dna_range, "dna_")

    df.columns = cols
    df.to_csv(output_csv, index=False)


def parse_range(arg):
    """
    Convert 'start end' or 'start' to a tuple (start, end or None)
    """
    if arg is None:
        return None
    parts = arg.split()
    if len(parts) == 1:
        return (int(parts[0]), None)
    elif len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    else:
        raise ValueError("Range must be one or two integers.")


def main():
    parser = argparse.ArgumentParser(description="Add morphology/env/dna prefixes to column ranges.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument("--morph_range", type=str, help="Range for morphology columns, e.g. '1 26' or '1'")
    parser.add_argument("--env_range", type=str, help="Range for environment columns")
    parser.add_argument("--dna_range", type=str, help="Range for DNA columns")

    args = parser.parse_args()

    add_prefix_by_index(
        args.input,
        args.output,
        morph_range=parse_range(args.morph_range),
        env_range=parse_range(args.env_range),
        dna_range=parse_range(args.dna_range)
    )

    print(f"\nDone! Output written to: {args.output}\n")

if __name__ == "__main__":
    main()
