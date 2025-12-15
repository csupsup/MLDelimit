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

#### Set missing data in FASTA sequence
```bash
## Randomly set missing nucleotides (either single or blocks)
python3 set_fasta_missing.py \
--input diploid_seq.fasta \
--output diploid_seq_masked.fasta \
--percent 10  --cblock 10 --limit 10 --single_prob 0.5
```

#### Set missing data in VCF file
```bash
## Ramdomly set missing genotype (e.g., 10%)
python3 set_vcf_missing.py \
--input s1_sampled_snps.vcf \
--output output.vcf \
--miss_genotype 0.10 \
--seed 1225
```

#### Convert VCF file to numeric genotype matrix
```bash
## Covert genotypes to numeric
python3 vcf_to_nmatrix.py \
--input s1_sampled_snps.vcf \
--output s1_gen_mat.csv
```