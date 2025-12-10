## Reconstruct Diploid Sequences

### Data needed

* SLiM outputs:
	1. ancestral_sequence.txt
	2. sampled_mutations.txt

To reconstruct diploid sequences of sampled individuals, use the custom python scripts above and run the following command in the terminal:

#### Reconstruct
```bash
## Reconstruct diploid sequence
python3 reconstruct_diploid_seq.py \
--ancestral ancestral_sequence.txt \
--mutations sampled_mutations.txt \
--output diploid_seq.fasta

```

#### Set missing data on full FASTA
```bash
## Set missing data on full fasta
python3 set_missing_data.py \
--input diploid_seq.fasta \
--output diploid_seq_masked.fasta \
--percent 10  --cblock 10 --limit 10 --single_prob 0.5
```

#### Remove invariant sites to get SNPs
```bash
## Remove invariant sites
python3 remove_invar_from_fasta.py \
--input diploid_seq.fasta \
--output diploid_snp.fasta

```

#### Set missing data on SNPs
```bash
python3 set_missing_data.py \
--input s2_final_data_08Dec2025/diploid_snp.fasta \
--output s2_final_data_08Dec2025/diploid_snp_masked.fasta \
--percent 10  --cblock 10 --limit 10 --single_prob 0.5

```