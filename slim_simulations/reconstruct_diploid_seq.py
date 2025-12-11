#! /usr/bin/env python3

from collections import defaultdict
import argparse

## IUPAC ambiguity codes for heterozygous sites
IUPAC = {
    frozenset(['A', 'G']): 'R',
    frozenset(['C', 'T']): 'Y',
    frozenset(['G', 'C']): 'S',
    frozenset(['A', 'T']): 'W',
    frozenset(['G', 'T']): 'K',
    frozenset(['A', 'C']): 'M'
}

def main():
    ## Command line arguments
    parser = argparse.ArgumentParser(
        description="Reconstruct diploid sequences of sampled individuals."
    )
    parser.add_argument("--ancestral", "-a", required=True,
                        help="Path to ancestral sequence file")
    parser.add_argument("--mutations", "-m", required=True,
                        help="Path to sampled mutations file")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output diploid FASTA file")
    
    args = parser.parse_args()

    ## Print argument values
    print("Arguments:")
    print(f"  ancestral : {args.ancestral}")
    print(f"  mutations : {args.mutations}")
    print(f"  output    : {args.output}\n")

    ## Load ancestral sequence
    with open(args.ancestral) as f:
        ancestral = list(f.read().strip())

    L = len(ancestral)

    ## Parse sampled mutations
    diploid_mutations = defaultdict(lambda: defaultdict(set))

    with open(args.mutations) as f:
        header = next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            sample_idx, pop, label, pos, derived, *rest = parts
            ind_key = f"{sample_idx}_{pop}"
            pos = int(pos)
            diploid_mutations[ind_key][pos].add(derived)

    ## Write diploid FASTA 
    with open(args.output, "w") as out:
        for ind, mut_dict in diploid_mutations.items():
            seq = ancestral.copy()
            for pos, alleles in mut_dict.items():
                if len(alleles) == 1:
                    seq[pos] = next(iter(alleles))
                elif len(alleles) == 2:
                    code = IUPAC.get(frozenset(alleles))
                    seq[pos] = code if code else 'N'
                else:
                    seq[pos] = 'N'
            out.write(f">{ind}\n")
            out.write("".join(seq) + "\n")
            
    print(f"Done! Output written to {args.output}")

if __name__ == "__main__":
    main()
