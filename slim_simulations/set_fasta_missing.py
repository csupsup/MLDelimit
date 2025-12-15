#!/usr/bin/env python3

import random
import argparse
from Bio import SeqIO
from Bio.Seq import Seq

def mask_blocks_and_singles(seq, percent, block_length, single_prob=0.5, mask_char='N'):
    """
    Mask sequence with:
    - Multiple continuous blocks of length "block_length"
    - Random single Ns
    Set block or single N is with "single_prob"
    """
    length = len(seq)
    total_to_mask = int(length * (percent / 100.0))
    total_to_mask = min(total_to_mask, length)

    masked_positions = set()
    while len(masked_positions) < total_to_mask:
        remaining = total_to_mask - len(masked_positions)
        if remaining >= block_length and random.random() > single_prob:
            
            ## Add continuous block
            start = random.randint(0, length - block_length)
            block_positions = set(range(start, start + block_length))
            masked_positions.update(block_positions - masked_positions)
        else:
            
            ## Add single N
            remaining_positions = list(set(range(length)) - masked_positions)
            if not remaining_positions:
                break
            masked_positions.add(random.choice(remaining_positions))

    ## Trim if overshot
    if len(masked_positions) > total_to_mask:
        masked_positions = set(random.sample(list(masked_positions), total_to_mask))

    ## Apply masking
    seq_list = list(str(seq))
    for i in masked_positions:
        seq_list[i] = mask_char

    return ''.join(seq_list)


def main():
    parser = argparse.ArgumentParser(
        description="Mask sequences with multiple blocks and single Ns."
    )
    parser.add_argument("--input", "-i", dest="input_fasta", required=True, help="Input FASTA file")
    parser.add_argument("--percent", type=float, required=True, help="Percentage of sequence to mask")
    parser.add_argument("--cblock", type=int, required=True, help="Length of continuous block of masking")
    parser.add_argument("--single_prob", type=float, default=0.5,
                        help="Probability of adding single Ns instead of a block (0-1, default=0.5)")
    parser.add_argument("--output", "-o", dest="output_fasta", required=True, help="Output FASTA file")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Randomly apply masking to N sequences only")

    args = parser.parse_args()

    ## Print arguments
    print("Arguments:")
    print(f"  input_fasta : {args.input_fasta}")
    print(f"  percent     : {args.percent}")
    print(f"  cblock      : {args.cblock}")
    print(f"  single_prob : {args.single_prob}")
    print(f"  output_fasta: {args.output_fasta}")
    print(f"  limit       : {args.limit}\n")

    ## Validate input
    if not (0 < args.percent <= 100):
        parser.error("percent must be between 0 and 100")
    if args.cblock < 1:
        parser.error("cblock must be at least 1")
    if not (0 <= args.single_prob <= 1):
        parser.error("single_prob must be between 0 and 1")

    records = list(SeqIO.parse(args.input_fasta, "fasta"))
    total = len(records)

    mask_indices = set()
    if args.limit is not None:
        if args.limit > total:
            parser.error(f"--limit cannot be greater than number of sequences in the file ({total})")
        mask_indices = set(random.sample(range(total), args.limit))

    with open(args.output_fasta, "w") as out_handle:
        for i, record in enumerate(records):
            if args.limit is None or i in mask_indices:
                masked_seq = mask_blocks_and_singles(
                    record.seq, args.percent, args.cblock, args.single_prob
                )
                record.seq = Seq(masked_seq)
            SeqIO.write(record, out_handle, "fasta")

    print(f"Done! Output written to {args.output_fasta}")


if __name__ == "__main__":
    main()
