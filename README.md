# PCA and Clustering – CSCC11 Machine Learning

This branch contains **Assignment 3**, which focuses on **unsupervised learning**:  
1. **Principal Component Analysis (PCA)** for dimensionality reduction, visualization, and reconstruction.  
2. **Clustering methods** (K-Means, Gaussian Mixture Models) for grouping unlabeled data.  

Both coding and theory are combined to illustrate the role of **representation, optimization, and data structure** in machine learning.

---

## Theory

### 1. Principal Component Analysis (PCA)
PCA seeks a low-dimensional subspace that captures the maximum variance of high-dimensional data:contentReference[oaicite:0]{index=0}.  

- **Objective**: minimize reconstruction error  
  \[
  \min_{W,b,\{x_i\}} \sum_i \| y_i - (Wx_i+b) \|^2, \quad W^TW=I
  \]  

- **Eigen-decomposition**:  
  PCA basis vectors are eigenvectors of the covariance matrix. The top-\(k\) eigenvectors capture most variance.  

- **Interpretations**:contentReference[oaicite:1]{index=1}:  
  - **Visualization**: project MNIST images into 2D to observe digit clusters.  
  - **Compression**: reconstruct digits with fewer components.  
  - **Handling Missing Data**: reconstruct missing pixels using projection matrices.  
  - **Probabilistic PCA (PPCA)**: models data as Gaussian with covariance \( WW^T+\sigma^2 I \).  

---

### 2. Lagrange Multipliers in PCA
The **maximum variance formulation** of PCA can be derived via **Lagrange multipliers**:contentReference[oaicite:2]{index=2}.  
We maximize projected variance subject to \(w^Tw=1\):  
\[
\max_w \frac{1}{N}\sum_i (w^T(y_i-\bar y))^2 \quad \text{s.t. } w^Tw=1
\]  
This yields the eigenvector equation \( Cw = \lambda w \), linking PCA to constrained optimization.

---

### 3. Clustering
Clustering groups data points without labels:contentReference[oaicite:3]{index=3}.  

#### **K-Means**
- Objective: minimize within-cluster squared error  
  \[
  E = \sum_{i,j} \ell_{ij} \| y_i - c_j \|^2
  \]  
- Algorithm: Lloyd’s iteration (assignment → update).  
- Sensitive to initialization, improved by **K-Means++**.  

#### **Gaussian Mixture Models (GMM)**
- Models data as a mixture of Gaussians:  
  \[
  p(y) = \sum_{j=1}^K m_j \, \mathcal{N}(y;\mu_j,C_j)
  \]  
- Uses **Expectation–Maximization (EM)** for parameter estimation.  
- Provides **soft assignments**, unlike hard K-Means clustering.  
- Generalizes K-Means when variances shrink to zero:contentReference[oaicite:4]{index=4}.  

---

## Coding Implementation

- **PCA.py**  
  - Implements eigen-decomposition of covariance.  
  - Visualizes cumulative explained variance.  
  - Reconstructs MNIST digits with different numbers of components.  
  - Demonstrates missing pixel reconstruction.  

- **Clustering.py**  
  - Applies **K-Means** and **GMM** to MNIST subsets.  
  - Visualizes segmentation results and cluster centers.  
  - Compares robustness under feature scaling and redundant features.  

---

## Conclusion

- **Dimensionality Reduction**:  
  ~245 components capture 90% of MNIST variance. Lower dimensions yield blurry reconstructions, higher dimensions risk overfitting noise.  

- **Missing Data Reconstruction**:  
  PCA can plausibly reconstruct occluded pixels; too many components degrade results.  

- **K-Means**:  
  Performs well on normalized features; initialization impacts quality.  

- **GMM**:  
  Better handles elongated/overlapping clusters with probabilistic assignments.  


The coding exercises reinforce the theoretical principles by showing how **representation choices** (number of components, distance metrics, probabilistic modeling) influence results in real datasets.
