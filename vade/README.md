## VaDE for Joint Species Delimitation

Our implementation of Variational Deep Embedding (VaDE) was adapted from Jiang et al. (2017). VaDE is a generative clustering method that uses a variational autoencoder as its core. Our implementation generally follows Jiang et al. (2017), except for the addition of auto-selection of the number of clusters to initialize VaDE clustering prior by applying Gaussian Mixture Models (GMM) on the standard variational autoencoders' latent space. Additionally, we included KL-Divergence annealing, which gradually increases the KL term weight to allow the decoder to learn the reconstruction first, then softly align the latent space with the GMM clustering prior.

Data preparation is similar to our iDEC, with data being standardized prior to analysis. The summary of our implementation is shown below.

#### Variational Autoencoder

