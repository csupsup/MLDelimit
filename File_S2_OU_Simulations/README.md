## OU Morphlogy/Trait Simulation

Ornstein-Uhlenbeck (OU) Model is expressed as:

$$
\mathbf{X}(t) = \mathbf{X}_0 e^{-\alpha t} + \boldsymbol{\theta} \left(1 - e^{-\alpha t}\right) + \boldsymbol{\varepsilon}
$$

$$
\boldsymbol{\varepsilon} \sim \mathcal{N}\Bigg(\mathbf{0}, \frac{\sigma^2}{2\alpha} \left(1 - e^{-2\alpha t}\right)\mathbf{I}\Bigg)
$$

where:

* $\mathbf{X}(t)$ is the trait value at time $t$ (or at a descendant node)
* $\mathbf{X}_0$ is the ancestral trait value
* $\alpha$ is the strength of selection toward the optimum  
* $\boldsymbol{\theta}$ is the optimal trait value  
* $\sigma^2$ is the rate of stochastic fluctuations (Brownian Motion)
* $t$ is the branch length (time from root to tip)  
* $\boldsymbol{\varepsilon}$ is the normally distributed stochastic term

### Data needed

* A guide tree in Newick format (from IQTree)

### OU morph simulation arguments
```bash
Options:
  --help            	Check arguments
  --input 				Input Newick tree file
  --output 				Output directory to write simulated traits
  --n_traits			Number of traits
  --alpha				Alpha (selection strength)
  --sigma				Sigma (rate of stochastic fluctuations)
  --root_state			Optional root state values (comma-separated)
  --theta				Optional OU optima θ values (comma-separated)
  --n_simulations 		Number of simulations to average over
  --show_labels         Show tip labels on PCA plot
```

To perform OU trait simulation on a phylogenetic tree, run the following command line in the terminal:

```bash
## OU morph simulation
## Output folder is automatically created by --ouput
python3 ou_morph.py \
--input s2_diploid_seq_masked.fasta.nwk \
--output ./outputs \
--n_traits 25 \
--alpha 0.5 \
--sigma 0.5 \
--n_simulations 5 \
--show_labels
```