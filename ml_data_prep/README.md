## ML Data Preparation

To perform ML analysis with iDEC, VaDE, and SOM, all data types must be combined into a CSV dataframe. The custom functions below demonstrate how to prepare the required data format. 

### Data Needed

* Morphology/Environment data in CSV file
* Genotype matrix in CSV file
* A popmap file containing the population assignment of each sample

### Concatenate all data types
```bash
python3 concat_data.py \
--input_1 s1_tip_traits.csv \
--input_2 s1_gen_mat.csv \
--output concat_matrix.csv
```

### Replace IDs with population label
```bash
python3 replace_ids.py \
--input_1 concat_matrix.csv \
--input_2 s1_popmap.csv \
--output concat_matrix_pop.csv
```

### Assign prefixes
```bash
## This function assigns prefixes like "morph_", "env_", and "dna_" to specified data type columns
## Prefixes serve as reference for ML modalities and ablation tests
python3 assign_prefix.py \
--input concat_matrix_pop.csv \
--output s1_concat_matrix_final.csv \
--morph_range "1 26" \
--dna_range "26"
```