#!/usr/bin/env python3

import argparse
import random
import sys
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Randomly assign missing genotypes in a VCF file")
    parser.add_argument("--input", required=True, help="Input VCF file")
    parser.add_argument("--output", required=True, help="Output VCF file")
    parser.add_argument("--miss_genotype", type=float, required=True, help="Proportion of genotypes to set as missing (0-1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    return parser.parse_args()

## Regex to check if genotype is valid (0,1,2,...) with / or | or missing
GENOTYPE_REGEX = re.compile(r"^(\d+|\.)[/|](\d+|\.)$")

def make_missing_genotype(genotype, miss_rate):
    ## Randomly replace entire genotype with './.' or '.|.' based on miss_rate
    
    ## If already missing, keep as is
    if genotype in {".", "./.", ".|."}:
        return genotype

    ## Randomly decide to make it missing
    if random.random() < miss_rate:
        sep = "/" if "/" in genotype else "|"
        return f".{sep}."

    ## Check valid genotype
    if not GENOTYPE_REGEX.match(genotype):
        sys.stderr.write(f"Warning: unexpected genotype format '{genotype}', keeping as is\n")
    return genotype


def main():
    args = parse_args()

    if not (0.0 <= args.miss_genotype <= 1.0):
        sys.exit("Error: --miss_genotype must be between 0 and 1")

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            fixed = fields[:9]  # CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT
            samples = fields[9:]

            new_samples = []
            for sample in samples:
                parts = sample.split(":")
                gt = parts[0]
                parts[0] = make_missing_genotype(gt, args.miss_genotype)
                new_samples.append(":".join(parts))

            fout.write("\t".join(fixed + new_samples) + "\n")

    print(f"Done! Output VCF written to: {args.output}")

if __name__ == "__main__":
    main()
