#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:04:49 2024

@author: mehdy
"""

################################################################################
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = np.asarray(mnist.data, dtype='float32')
y = np.asarray(mnist.target, dtype = 'int32')

x_train  = X[:60000]
y_train = y[:60000]
x_test = X[60000:]
y_test = y[60000:]

# Normalize the data
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train) # Use a scaler to normalize the x_train data
x_test_scale =  scaler.transform(x_test) # Use scaler with respect to the mean and standard deviation derived from x_train to normalize x_test

pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_train_scale) # find Principal components of train data


# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=matplotlib.colormaps['tab10'], s=5)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.get_cmap('tab10', 10), s=5)
plt.colorbar(label='Digit', ticks=range(10))
plt.clim(-0.5, 9.5)
plt.title('PCA Visualization of MNIST')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

#########################################################################################
pca = PCA(n_components=500) 
X_pca = pca.fit_transform(x_train_scale) #find Principal components of train data
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_) # find variance of each comp
# Plot cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(cumulative_variance_ratio, marker='o', linestyle='-')
plt.xlabel('Number of PC')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.show()

#######################################################################################
n_components_reconstruction = 500
pca = PCA(n_components=n_components_reconstruction) 
X_pca = pca.fit_transform(x_train_scale) #find Principal components of train data

X_reconstructed = pca.inverse_transform(X_pca) # Reconstruct input images using the first 50,100, and 500 principal components 
X_reconstructed = X_reconstructed *scaler.scale_ + scaler.mean_
X_reconstructed = X_reconstructed.reshape(-1, 28, 28)

plt.imshow(X_reconstructed[0],cmap='gray')
#######################################################################################

pca = PCA(n_components=500)
pca.fit(x_train_scale)
def create_P(height=28, width=28, pattern="bottom"):
    if pattern == "bottom":
        m = height * width // 2
        P = np.zeros((m, height*width))
        P[:m, :m] = np.eye(m)
    return P

pattern = "bottom"
W = pca.components_.T
P = create_P(pattern=pattern)
projected_partial_obs = P @ x_test_scale.T # projection to subspace (m x n)
zero_filled_obs = P.T @ P @ x_test.T # (n x n)


Leas_square_solution_x = np.linalg.inv(W.T @ P.T @ P @ W) @ W.T @ P.T @ projected_partial_obs #Least square solution discussed in 2.b
reconst = W @ Leas_square_solution_x # Reconstcut the input image using Leas_square_solution_x and W. 
rescaled_reconst = reconst * scaler.scale_[:, None] + scaler.mean_[:, None]


idx = 130
reshaped_reconst = rescaled_reconst.T.reshape(-1, 28, 28)
reshaped_partial_obs = zero_filled_obs.T.reshape(-1, 28, 28)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(reshaped_partial_obs[idx], cmap='gray')
ax[1].imshow(reshaped_reconst[idx], cmap='gray')
plt.show()


