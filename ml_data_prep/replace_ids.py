#!/usr/bin/env python3

import pandas as pd
import argparse

## Replace IDs with the corresponding population label
def replace_id_with_pop(file1_path, file2_path, output_path):
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    if "ID" not in df1.columns:
        raise ValueError(f"Error: 'ID' column not found in {file1_path}")
    if "ID" not in df2.columns or "Pop" not in df2.columns:
        raise ValueError(f"Error: 'ID' and 'Pop' columns are required in {file2_path}")

    id_to_pop = dict(zip(df2["ID"], df2["Pop"]))

    missing_ids = []
    def map_id(x):
        if x not in id_to_pop:
            missing_ids.append(x)
            return x
        return id_to_pop[x]

    df1["ID"] = df1["ID"].apply(map_id)
    df1.to_csv(output_path, index=False)

    ## Summary
    print(f"\nDone! Output written to: {output_path}\n")

    if missing_ids:
        print("Some IDs were NOT found in input_2 (popmap):")
        for mid in sorted(set(missing_ids)):
            print("   -", mid)
    else:
        print("All IDs successfully replaced with population labels!")

def main():
    parser = argparse.ArgumentParser(
        description="Replace IDs in a CSV with population labels using a popmap file.")
    parser.add_argument(
        "--input_1", "-i1", required=True, help="Input CSV file with IDs to replace.")
    parser.add_argument(
        "--input_2", "-i2", required=True, help="Population map CSV file containing columns 'ID' and 'Pop'.")
    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file with updated population labels.")

    args = parser.parse_args()

    replace_id_with_pop(args.input_1, args.input_2, args.output)


if __name__ == "__main__":
    main()
