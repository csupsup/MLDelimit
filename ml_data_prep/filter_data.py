#! /usr/bin/env python3

import argparse
import pandas as pd
from Bio import SeqIO

def filter_csv_and_fasta(csv_file, fasta_file, id_column, output_csv, output_fasta):
    df = pd.read_csv(csv_file)
    csv_ids = set(df[id_column].astype(str).tolist())
    fasta_records = list(SeqIO.parse(fasta_file, "fasta"))
    fasta_ids = set(record.id for record in fasta_records)

    matching_ids = csv_ids.intersection(fasta_ids)

    ## Print filtering summary
    only_in_csv = csv_ids - fasta_ids
    only_in_fasta = fasta_ids - csv_ids

    print(f"\nFiltering Summary")
    print(f"Total IDs in CSV:   {len(csv_ids)}")
    print(f"Total IDs in FASTA: {len(fasta_ids)}")
    print(f"Matching IDs:       {len(matching_ids)}")

    if only_in_csv:
        print(f"\nIDs present in CSV but NOT in FASTA ({len(only_in_csv)}):")
        for x in sorted(only_in_csv):
            print(f"  - {x}")

    if only_in_fasta:
        print(f"\nIDs present in FASTA but NOT in CSV ({len(only_in_fasta)}):")
        for x in sorted(only_in_fasta):
            print(f"  - {x}")

    if not only_in_csv and not only_in_fasta:
        print("\nAll IDs match between CSV and FASTA")

    ## Filtering
    filtered_df = df[df[id_column].astype(str).isin(matching_ids)]
    filtered_fasta_records = [rec for rec in fasta_records if rec.id in matching_ids]

    filtered_df.to_csv(output_csv, index=False)
    SeqIO.write(filtered_fasta_records, output_fasta, "fasta")

    print(f"\nDone! Filtered CSV written to: {output_csv}")
    print(f"Done! Filtered FASTA written to: {output_fasta}\n")

## Command line arguments
def main():
    parser = argparse.ArgumentParser(
        description="Filter CSV and FASTA files to keep only matching sequence/sample IDs."
    )

    parser.add_argument("--input_csv", required=True,
                        help="Input CSV file containing morphology/environmenta data.")
    parser.add_argument("--input_fasta", required=True,
                        help="Input FASTA file containing sequences.")
    parser.add_argument("--id_column", required=True,
                        help="Column name in the CSV containing sample IDs.")
    parser.add_argument("--output_csv", required=True,
                        help="Output CSV file after filtering.")
    parser.add_argument("--output_fasta", required=True,
                        help="Output FASTA file after filtering.")

    args = parser.parse_args()

    filter_csv_and_fasta(
        args.input_csv,
        args.input_fasta,
        args.id_column,
        args.output_csv,
        args.output_fasta
    )


if __name__ == "__main__":
    main()
