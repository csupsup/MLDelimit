#!/usr/bin/env python3

import argparse

def read_fasta(filename):
    seqs = {}
    with open(filename) as f:
        name = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                name = line[1:].strip()
                seqs[name] = []
            else:
                seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}

def write_fasta(seqs, out):
    with open(out, "w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n")
            f.write(seq + "\n")

def remove_invariant(seqs):
    names = list(seqs.keys())
    aligned = [seqs[n] for n in names]

    L = len(aligned[0])
    keep_cols = []

    for i in range(L):
        column = {seq[i] for seq in aligned}
        if len(column) > 1:
            keep_cols.append(i)

    trimmed = {}
    for name in names:
        trimmed[name] = "".join(seqs[name][i] for i in keep_cols)
    return trimmed

def main():
    parser = argparse.ArgumentParser(
        description="Remove invariant sites from FASTA sequences."
    )
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file")
    parser.add_argument("--output", "-o", required=True, help="Output FASTA file")
    args = parser.parse_args()

    # Print parsed arguments
    print("Arguments:")
    print(f"  input  : {args.input}")
    print(f"  output : {args.output}\n")

    seqs = read_fasta(args.input)

    # Sanity check: all sequences same length
    lengths = {len(s) for s in seqs.values()}
    if len(lengths) != 1:
        raise ValueError("Error: sequences are not aligned or not equal length.")

    trimmed = remove_invariant(seqs)
    write_fasta(trimmed, args.output)

    print(f"Done! Output written to {args.output}")

if __name__ == "__main__":
    main()