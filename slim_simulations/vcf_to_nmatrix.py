#!/usr/bin/env python3

import argparse
import pysam
import pandas as pd
from collections import Counter

pysam.set_verbosity(0)

## Allele conversion values
SNV_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
SNV_MISSING = 4

## Parse functions
def is_snv(ref, alts):
    if len(ref) != 1:
        return False
    for a in alts:
        if a is None or len(str(a)) != 1:
            return False
    return True

def parse_gt(gt_tuple):
    if gt_tuple is None:
        return [None, None]
    return [g if g is not None else None for g in gt_tuple]

def encode_snv(gt, alleles, stats):
    out = []
    for g in gt:
        stats["total_alleles"] += 1
        if g is None:
            out.append(SNV_MISSING)
            stats["missing_alleles"] += 1
        else:
            allele = alleles[g].upper()
            if allele not in SNV_MAP:
                out.append(SNV_MISSING)
                stats["non_acgt_alleles"][allele] += 1
            else:
                out.append(SNV_MAP[allele])
    return out

def encode_indel(gt, ref, alts, stats):
    out = []
    for g in gt:
        stats["total_alleles"] += 1
        if g is None:
            out.extend([0, 0])
            stats["missing_alleles"] += 1
        elif g == 0:
            out.extend([0, 0])
        else:
            alt = alts[g - 1]
            out.extend([1, len(alt) - len(ref)])
    return out

## Main function
def main(vcf_path, out_csv):
    vcf = pysam.VariantFile(vcf_path)

    samples = list(vcf.header.samples)
    rows = {s: [] for s in samples}
    columns = []

    ## Store stats
    stats = {
        "snv_records": 0,
        "indel_records": 0,
        "missing_alleles": 0,
        "total_alleles": 0,
        "phased_detected": False,
        "non_acgt_alleles": Counter(),
    }

    for rec in vcf:
        ref = rec.ref
        alts = list(rec.alts) if rec.alts else []

        ## Skip symbolic alleles
        if any(a.startswith("<") or a == "*" for a in alts):
            continue

        chrom = rec.chrom
        pos = rec.pos

        ## Detect phasing
        if not stats["phased_detected"]:
            for s in samples:
                if rec.samples[s].phased:
                    stats["phased_detected"] = True
                    break

        if is_snv(ref, alts):
            stats["snv_records"] += 1

            columns.extend([
                f"{chrom}_{pos}_SNV_a1",
                f"{chrom}_{pos}_SNV_a2"
            ])

            alleles = [ref] + alts
            for s in samples:
                gt = parse_gt(rec.samples[s]["GT"])
                rows[s].extend(encode_snv(gt, alleles, stats))

        else:
            stats["indel_records"] += 1

            columns.extend([
                f"{chrom}_{pos}_INDEL_f1",
                f"{chrom}_{pos}_INDEL_l1",
                f"{chrom}_{pos}_INDEL_f2",
                f"{chrom}_{pos}_INDEL_l2",
            ])

            for s in samples:
                gt = parse_gt(rec.samples[s]["GT"])
                rows[s].extend(encode_indel(gt, ref, alts, stats))

    ## Build dataframe
    df = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    df.insert(0, "ID", df.index)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(out_csv, index=False)

    ## Report
    print("\n--- Genotype Conversion Summary ---")
    print(f"Input VCF                  : {vcf_path}")
    print(f"Samples                    : {len(samples)}")
    print(f"SNV records                : {stats['snv_records']}")
    print(f"INDEL records              : {stats['indel_records']}")
    print(f"Total alleles              : {stats['total_alleles']}")
    print(f"Missing alleles (.)        : {stats['missing_alleles']}")

    if stats["total_alleles"] > 0:
        rate = stats["missing_alleles"] / stats["total_alleles"]
        print(f"Missing allele rate        : {rate:.4f}")

    if stats["non_acgt_alleles"]:
        non_acgt_str = ", ".join(
            f"{a} ({c})" for a, c in stats["non_acgt_alleles"].items()
        )
    else:
        non_acgt_str = "None"

    print(f"Non-ACGT SNV alleles       : {non_acgt_str}")
    print(f"Phased genotypes detected  : {'YES' if stats['phased_detected'] else 'NO'}")
    print(f"Output CSV                 : {out_csv}")
    print("---\n")

    print(f"Done! Coverted file written to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VCF to numeric genotype matrix")
    parser.add_argument("--input", required=True, help="Input VCF file")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()
    main(args.input, args.output)


