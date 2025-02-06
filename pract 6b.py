# pip install numpy matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data for clustering
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model
num_components = 3  # Number of Gaussian components
GMM = GaussianMixture(n_components=num_components, covariance_type='full', random_state=42)
GMM.fit(X)
labels = GMM.predict(X)

# Create a grid for density estimation
x, y = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
                    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
XY = np.array([x.ravel(), y.ravel()]).T
density = np.exp(GMM.score_samples(XY)).reshape(x.shape)

# Plot combined clustering and density estimation
plt.figure(figsize=(8, 6))

# Density estimation contour
plt.contourf(x, y, density, levels=15, cmap='Blues', alpha=0.6)

# Scatter plot for clustering
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)

plt.title("Gaussian Mixture Model: Clustering and Density Estimation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Density')
plt.show()
