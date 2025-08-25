# Regression Branch – CSCC11 Machine Learning

This branch focuses on **regression methods** as studied in *CSCC11 – Introduction to Machine Learning (University of Toronto)*.  
The central theme is how different **basis functions** can be used for regression, their theoretical foundation, and their applications.

---

## Theory

### 1. Polynomial Regression
Polynomial regression extends linear regression by mapping the input \(x\) into a higher-dimensional polynomial space:
\[
f(x) = b + w_1 x + w_2 x^2 + \dots + w_K x^K
\]

- **Least Squares Solution**:  
  Parameters are obtained by minimizing the mean squared error (MSE):  
  \[
  \hat{w} = (B^T B)^{-1} B^T y
  \]
  where \(B\) is the design matrix built from polynomial basis functions.

- **Regularization (Ridge Regression)**:  
  To prevent overfitting with high-degree polynomials, an L2 penalty is introduced:  
  \[
  \hat{w} = (B^T B + \lambda I)^{-1} B^T y
  \]
  This shrinks coefficients and improves generalization.

- **Conclusion**:  
  Polynomial regression illustrates the bias-variance tradeoff:
  - Small \(K\): underfitting (high bias).  
  - Large \(K\): overfitting (high variance).  
  - Proper choice of \(K\) and \(\lambda\): balance between them.

---

### 2. Radial Basis Function (RBF) Regression
RBF regression uses localized Gaussian basis functions in (usually) higher-dimensional spaces:
\[
f(x) = b + \sum_{i=1}^K w_i \exp\left(-\frac{\|x - c_i\|^2}{2\sigma_i^2}\right)
\]

- **Centers & Widths**:  
  Each basis function is centered at \(c_i\) with width \(\sigma_i\), controlling locality.

- **Least Squares with Regularization**:  
  As with polynomials, parameters are estimated via regularized least squares to avoid overfitting.

- **Applications**:  
  RBF regression is powerful for non-linear problems and tasks with noisy data.  
  In this coding, it is applied to **image denoising**: noisy pixels are smoothed by fitting local Gaussian functions, preserving structure while reducing noise.


- RBFs adapt locally and handle noise well, but require careful choice of centers and widths.  
- Both methods demonstrate the core idea: **linear regression in feature space**, where the features are defined by the chosen basis functions.
