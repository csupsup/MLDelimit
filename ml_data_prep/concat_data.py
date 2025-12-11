#!/usr/bin/env python3

import csv
import argparse

## Concatenate morphology/environment and DNA (one-hot) data
def merge_csv_files(file1, file2, output):
    def clean_column_name(name):
        return name.strip()

    ## Read DNA one-hot
    with open(file2, newline='') as f2:
        reader2 = csv.reader(f2)
        header2 = next(reader2)
        header2 = [clean_column_name(col) for col in header2]

        id_column_index = 0
        f2_data = {}

        for row in reader2:
            row_id = row[id_column_index]
            f2_data[row_id] = row[1:]

    ## Read morph/env and write concatenated output
    with open(file1, newline='') as f1, open(output, 'w', newline='') as out:
        reader1 = csv.reader(f1)
        header1 = next(reader1)
        header1 = [clean_column_name(col) for col in header1]

        new_header = header1 + header2[1:]
        writer = csv.writer(out)
        writer.writerow(new_header)

        missing_in_file2 = []

        for row1 in reader1:
            row_id = row1[0]
            if row_id in f2_data:
                row1.extend(f2_data[row_id])
            else:
                row1.extend([''] * (len(header2) - 1))
                missing_in_file2.append(row_id)
            writer.writerow(row1)

    ## Concatenation summary
    if missing_in_file2:
        print("\nThe following IDs were NOT found in input_2:")
        for mid in missing_in_file2:
            print("   -", mid)
    else:
        print("\nAll IDs matched successfully!")

    print(f"\nDone! Concatenated data written to: {output}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate morphology/environment data with DNA (one-hot).")
    parser.add_argument(
        "--input_1", "-i1", required=True, help="First input CSV file (e.g., morphology or environment data)")
    parser.add_argument(
        "--input_2", "-i2", required=True, help="Second input CSV file (DNA one-hot encoded data)")
    parser.add_argument(
        "--output", "-o", required=True, help="Output concatenated CSV file")

    args = parser.parse_args()

    merge_csv_files(args.input_1, args.input_2, args.output)


if __name__ == "__main__":
    main()
