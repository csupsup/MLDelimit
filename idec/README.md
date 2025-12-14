## iDEC for Joint Species Delimitation

### Overview

We developed two implementations of Improved Deep Embedded Clustering (iDEC) adapted from Guo et al. (2017) to simultaneously learn low-dimensional representations and clustering assignments from multiple data types, such as genetic, morphological, and environmental data. A single-modal iDEC uses only one neural network architecture, where all data types are compressed into a single latent representation. A multi-modal iDEC features three neural network architectures, meaning it has separate encoders and decoders for each data type, with each latent representation concatenated. The latent representations are learned jointly with soft clustering optimized via a KL-divergence-based clustering loss. Additionally, we included an ablation comparison to evaluate the clustering results between pairs of data types using the Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI) metrics.

Below shows the summary of the implementation.

### Single-modal iDEC (iDEC-Single)
#### Data preparation
Let the input data matrix be:
$$
\mathbf{X} \in \mathbb{R}^{N \times D}
$$

where
* $\textit{N}$ is the number of samples.
* $\textit{D}$ is the total number of features across all data types.

Each feature is standardized as:
$$
\mathbf{X} \leftarrow \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$

* $\mathbf{X} \in \mathbb{R}^{N \times D}$ is the input feature matrix (samples $\times$ features).
* $\boldsymbol{\mu} \in \mathbb{R}^{1 \times D}$ is the mean vector of each feature.  
* $\boldsymbol{\sigma} \in \mathbb{R}^{1 \times D}$ is the standard deviation vector of each feature.
* $N$ is the number of samples, and $D$ is the number of features.

For the ablation study, missing modalities are zero-masked while keeping the dimension fixed.

#### Autoencoder Architecture
The model uses a fully connected autoencoder where the encoder maps each input sample to a latent representation:
$$
\mathbf{z}_i = f_{\theta}(\mathbf{x}_i), \quad \mathbf{z}_i \in \mathbb{R}^{L}
$$

* $\mathbf{x}_i \in \mathbb{R}^{D}$ is the input feature vector for sample $i$. 
* $f_{\theta}(\cdot)$ is the encoder function implemented as a deep neural network.
* $\theta$ are the trainable parameters of the encoder.
* $\mathbf{z}_i \in \mathbb{R}^{L}$ is the latent representation of sample $i$ and $L$ is the dimensionality of the latent space.

then, the decoder reconstructs the input from the latent space:
$$
\hat{\mathbf{x}}_i = g_{\phi}(\mathbf{z}_i)
$$

* $g_{\phi}(\cdot)$ is the decoder function (deep neural network).
* $\phi$ are the trainable parameters of the decoder
* $\hat{\mathbf{x}}_i \in \mathbb{R}^{D}$ is the reconstructed input of sample $i$.

with mean squared error (MSE) reconstruction loss:
$$
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \left\| \hat{\mathbf{x}}_i - \mathbf{x}_i \right\|_2^2
$$

* $\mathcal{L}_{\text{recon}}$ is the reconstruction loss.
* $N$ is the number of samples
* $\hat{\mathbf{x}}_i$ is the reconstructed input, and $\mathbf{x}_i$ is the original input.

#### K-means initialization
Cluster centers are initialized with K-means:
$$
\{ \boldsymbol{\mu}_j \}_{j=1}^{K} = \text{KMeans}(\{ \mathbf{z}_i \}_{i=1}^{N})
$$

* $K$ is the number of clusters.
* $\mathbf{Z} = \{\mathbf{z}_i\}_{i=1}^N$ are latent embeddings.
* $\boldsymbol{\mu}_j \in \mathbb{R}^{L}$ is the center of cluster $j$.

#### Soft Cluster Assignment
Soft assignments are computed using the Student's t-distribution:
$$
q_{ij} = \frac{\left(1 + \frac{\| \mathbf{z}_i - \boldsymbol{\mu}_j \|^2}{\alpha} \right)^{-\frac{\alpha+1}{2}}}{\sum_{j'=1}^{K} \left(1 + \frac{\| \mathbf{z}_i - \boldsymbol{\mu}_{j'} \|^2}{\alpha} \right)^{-\frac{\alpha+1}{2}}}
$$

* $q_{ij}$ is the soft assignment probability of sample $i$ to cluster $j$. 
* $\mathbf{z}_i$ is the latent embedding. 
* $\boldsymbol{\mu}_j$ is the cluster center.
* $\alpha$ is the degree of freedom parameter (default 1.0)

and the t-distribution is sharpened:
$$
p_{ij} = \frac{\frac{q_{ij}^2}{\sum_{i=1}^{N} q_{ij}}}{\sum_{j'=1}^{K} \frac{q_{ij'}^2}{\sum_{i=1}^{N} q_{ij'}}}
$$

* $p_{ij}$ is the target distribution of sample $i$ for cluster $j$.
* $q_{ij}$ is the soft assignment probability.

with KL-Divergence clustering loss:
$$
\mathcal{L}_{\text{cluster}} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{K} p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
$$

* $\mathcal{L}_{\text{cluster}}$ is the clustering loss.
* $p_{ij}$ is the target distribution.
* $q_{ij}$ is the soft assignment.

#### iDEC total loss
Final iDEC optimization is:
$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \gamma \, \mathcal{L}_{\text{cluster}}
$$

* $\mathcal{L}$ is the total loss.
* $\mathcal{L}_{\text{recon}}$ is the reconstruction loss.
* $\mathcal{L}_{\text{cluster}}$ is the clustering loss.
* $\gamma$ is the weight of the clustering loss.

#### Ablation metrics
Metrics include:
$$
\text{ARI} = \frac{\sum_{ij} \binom{n_{ij}}{2} - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right]/\binom{N}{2}}{\frac{1}{2} \left[\sum_i \binom{a_i}{2} + \sum_j \binom{b_j}{2}\right] - \left[\sum_i \binom{a_i}{2} \sum_j \binom{b_j}{2}\right]/\binom{N}{2}}
$$

$$
\text{NMI} = \frac{2 I(\mathbf{y}_{\text{full}}; \mathbf{y}_{\text{ablation}})}{H(\mathbf{y}_{\text{full}}) + H(\mathbf{y}_{\text{ablation}})}
$$

* $\text{ARI}(\mathbf{y}_{\text{full}}, \mathbf{y}_{\text{ablation}})$ is the Adjusted Rand Index comparing full and ablation clusterings.
* $\text{NMI}(\mathbf{y}_{\text{full}}, \mathbf{y}_{\text{ablation}})$ is the Normalized Mutual Information.
* $n_{ij}$ is the contingency table entry, $a_i$ and $b_j$ are cluster sums, and $N$ is the number of samples.

### Multi-modal iDEC (iDEC-Multi)
The implementation of iDEC-Multi is the same as iDEC-Single, except for multiple modalities where each is mapped to a latent space using the following encoder:
$$
\mathbf{z}_i^m = f_{\theta_m}(\mathbf{x}_i^m)
$$

* $\mathbf{x}_i^m \in \mathbb{R}^{D_m}$ is the input vector for sample $i$ in modality $m$
* $f_{\theta_m}(\cdot)$ is the modality-specific encoder.
* $\theta_m$ are its trainable parameters.
* $\mathbf{z}_i^m \in \mathbb{R}^{L_m}$ is the latent embedding.
* $L_m$ is the latent dimension per modality.

with the latent embeddings from all modalities are concatenated:
$$
\mathbf{z}_i = [\mathbf{z}_i^1, \mathbf{z}_i^2, \dots, \mathbf{z}_i^M] \in \mathbb{R}^{L}
$$

* $\mathbf{z}_i \in \mathbb{R}^{L}$ is the concatenated latent vector for sample $i$ across all modalities.
* $\mathbf{z}_i^m \in \mathbb{R}^{L_m}$ is the latent embedding of sample $i$ for modality $m$.
* $M$ is the total number of modalities.
* $L = \sum_{m=1}^{M} L_m$ is the total latent dimension (sum of modality-specific latent dimensions).

decoded with:
$$
\hat{\mathbf{x}}_i^m = g_{\phi_m}(\mathbf{z}_i)
$$

* $\hat{\mathbf{x}}_i^m \in \mathbb{R}^{D_m}$ is the reconstructed input vector for sample $i$ in modality $m$.
* $g_{\phi_m}(\cdot)$ is the modality-specific decoder function.
* $\phi_m$ are the trainable parameters of the decoder.
* $\mathbf{z}_i \in \mathbb{R}^{L}$ is the concatenated latent vector of sample $i$ across all modalities.

and minimizes the MSE across all modalities:
$$
\mathcal{L}_{\mathrm{recon}} = \sum_{m=1}^{M} \frac{1}{N} \sum_{i=1}^{N} \| \hat{\mathbf{x}}_i^m - \mathbf{x}_i^m \|_2^2
$$

* $N$ is the number of samples.
* $M$ is the number of modalities.
* $\mathbf{x}_i^m \in \mathbb{R}^{D_m}$ is the original input vector for sample $i$ in modality $m$.
* $\hat{\mathbf{x}}_i^m \in \mathbb{R}^{D_m}$ is the reconstructed input vector from the decoder.

### Using iDEC
**iDEC-Single**
```bash
Options:
  --input				Input CSV containing morphology, environment, and dna data.
  --plots_dir 			Directory to write plots.
  --outputs_dir 		Directory to write latent space, cluster assingments, and other outputs.
  --latent_dim			Latent dimension.
  --n_iter				Number of iDEC training iterations.
  --update_interval 	Update interval for cluster centers.
  --alpha				Alpha parameter for soft assignment.
  --gamma				GAMMA clustering loss weight.
  --learning_rate		Learning rate.
  --batch_size			Batch size.
  --k_min				Minimum K to test.
  --k_max				Maximum K to test.
  --seed				Random seed for reproducibility.
```
Sample use
```bash
python3 idec_single_delimit.py \
--input gono_concat_matrix_final.csv \
--plots_dir plots \
--outputs_dir outputs \
--latent_dim 10 \
--n_iter 50 \
--update_interval 10 \
--alpha 1.0 \
--gamma 1.0 \
--learning_rate 0.0001 \
--batch_size 16 \
--k_min 2 \
--k_max 5
```

**iDEC-Multi**
```bash
Options:
  --input		        Input CSV containing morphology, environment, and dna.
  --plots_dir			Directory to write plots.
  --outputs_dir			Directory to write latent space, cluster assingments, and other outputs.
  --latent_per_mod		Latent dimension per modality.
  --n_iter				Number of iDEC training iterations.
  --update_interval		Update interval for cluster centers.
  --gamma				GAMMA clustering loss weight.
  --alpha				Alpha parameter for soft assignment.
  --learning_rate		Learning rate.
  --batch_size			Batch size.
  --k_min				Minimum K to test.
  --k_max				Maximum K to test.
  --seed				Random seed for reproducibility.
```
Sample use
```bash
python3 idec_multi_modal.py \
--input seal_concat_matrix_final.csv \
--plots_dir plots \
--outputs_dir outputs \
--latent 8 \
--n_iter 50 \
--update_interval 10 \
--gamma 0.1 \
--alpha 1.0 \
--learning_rate 0.01 \
--batch_size 16 \
--k_min 2 \
--k_max 5 \
--wrap_width 30
```

### References
Guo, X., L. Gao, X. Liu, and J. Yin. 2017. Improved Deep Embedded Clustering with Local Structure Preservation. Pp. 1753–1759 in Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence. International Joint Conferences on Artificial Intelligence Organization, Melbourne, Australia.
