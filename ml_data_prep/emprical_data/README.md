## Emperical Data

Our empirical data comprised two genera of Agamid Lizards from the Philippines: Broncohcela and Gonocephalus. Bronchocela includes 73 samples representing four species (*B*. *carestiai*, *B*. *mallarii*, *B*. *marmorata*, and *B*. *philippina*), while Gonocephalus features 53 samples from two species (*G*. *sophiae* and *G*. *semperi*). The dataset includes morphology, environmental variables, and genomic SNPs. Morphological and genomic data were obtained from Supsup et al. (in review), with climatic data extracted from WorldClim (https://worldclim.org/data/worldclim21.html). Additionally, we incorporated data from Pyron et al. (2023) and Pyron 2023 on the Seal Salamander from eastern United States, previously used for ML-based species delimitation.

The following demonstrates the conversion of a genomic SNP VCF file to FASTA format, the body-size correction for morphology data, and a collinearity test for environmental variables.

### Data Needed

* Morphology/Environment data in CSV file
* Filtered genomic SNPS in VCF file
* A popmap file containing the population assignment of each sample

### Convert VCF to FASTA
```bash
## Use the 'vcf2phylip' tool from https://github.com/edgardomortiz/vcf2phylip
python3 vcf2phylip.py \
--input gono_filtered_75snps.vcf \
--phylip-disable \
--fasta \
--output-folder .
```

### Correct morphology data (in R)
```r
## Get GroupStruct from https://github.com/chankinonn/GroupStruct
library(GroupStruct)
library(dplyr)

## Read data
morph <- read.csv("gono_morph.csv")

## Subset mensural data
sub_morph <- morph[2:18]

## Correct mensural data
## Note that the allom function shuffles the samples, so be careful when merging the output with other data
trans_morph <- allom(sub_morph, type = "population2")

## Merge back with meristic data
mer_morph <- subset(morph, select = c(2, 19:39))
merged_morph <- trans_morph %>%
  				full_join(mer_morph, by = "ID")

## Write
write.csv(merged_morph, "gono_morph_cor.csv", row.names = FALSE)
```

### Perform collinearity test with environmental variables
```r
library(corrplot)
library(dplyr)

env <- read.csv("gono_env.csv")
env$RowID <- NULL
env$ID <- NULL

cor_mat <- cor(env, use = "pairwise.complete.obs", method = "pearson")

## Plot
corrplot(
  cor_mat,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45
)

## Check pair
high_corr <- which(abs(cor_mat) > 0.8 & abs(cor_mat) < 1, arr.ind = TRUE)

high_corr_pairs <- data.frame(
  var1 = rownames(cor_mat)[high_corr[,1]],
  var2 = colnames(cor_mat)[high_corr[,2]],
  r    = cor_mat[high_corr]
)

high_corr_pairs

## Remove variables
env <- read.csv("gono_env.csv")
env$RowID <- NULL
env <- env %>% select(-bio_14, -bio_17, -bio_16, -bio_10, -bio_11, -bio_5, -bio_6, -bio_8, -bio_9)

write.csv(env, "gono_env_reduced.csv", row.names = FALSE)
```

```r
## Merge morphology and environment data
morph <- read.csv("gono_morph_cor.csv")
env <- read.csv("gono_env_reduced.csv")

merged_df <- morph %>%
  				full_join(env, by = "ID")
  				
write.csv(merged_df, "gono_morph_env.csv", row.names = FALSE)
```

### References

Pyron, R. 2023. Unsupervised machine learning for species delimitation, integrative taxonomy, and biodiversity conservation. Molecular Phylogenetics and Evolution 189:107939. DOI: https://doi.org/10.1016/j.ympev.2023.107939

Pyron, R.A., K.A. O’Connell, S.C. Duncan, F.T. Burbrink, and D.A. Beamer. 2023. Speciation Hypotheses from Phylogeographic Delimitation Yield an Integrative Taxonomy for Seal Salamanders (Desmognathus monticola). Systematic Biology 72:179–197. DOI: https://doi.org/10.1093/sysbio/syac065


