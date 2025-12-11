#!/usr/bin/env python3

import csv
import argparse
from Bio import SeqIO

## Get unique characters across sequences
def get_unique_characters(fasta_file):
    unique_chars = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        unique_chars.update(str(record.seq).upper())
    return sorted(unique_chars)

## Map each character to one-hot vector
def create_one_hot_dict(unique_chars):
    return {char: [1 if i == idx else 0 for i in range(len(unique_chars))]
            for idx, char in enumerate(unique_chars)}

## Encode sequence into one-hot
def encode_sequence(seq, one_hot_dict, width):
    return [one_hot_dict.get(char.upper(), [0] * width) for char in seq]

## Convert FASTA to one-hot
def fasta_to_one_hot_csv(fasta_file, output):
    unique_chars = get_unique_characters(fasta_file)
    one_hot_dict = create_one_hot_dict(unique_chars)
    width = len(unique_chars)

    with open(output, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        max_seq_length = max(len(record.seq) for record in SeqIO.parse(fasta_file, "fasta"))
        
        header = ['ID'] + [f'{pos}_{char}' for pos in range(max_seq_length) for char in unique_chars]
        writer.writerow(header)

        for record in SeqIO.parse(fasta_file, "fasta"):
            encoded = encode_sequence(str(record.seq), one_hot_dict, width)
            flat = [bit for vec in encoded for bit in vec]
            row = [record.id] + flat
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a FASTA file to a one-hot encoded CSV.")
    parser.add_argument(
        "--input", "-i", required=True, help="Input FASTA file.")
    parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file for one-hot encoded sequences.")

    args = parser.parse_args()

    fasta_to_one_hot_csv(args.input, args.output)

    print(f"\nDone! One-hot written to: {args.output}\n")


if __name__ == "__main__":
    main()
