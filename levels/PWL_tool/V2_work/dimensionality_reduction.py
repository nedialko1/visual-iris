import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from etc_utils.data_harvest_io import load_uci_data

# Load and Prep
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)

X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
components = pca.fit_transform(X_std)
vaf = pca.explained_variance_ratio_ * 100

# Print PC Loadings to Console
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=features)
print("\n--- Principal Component Loadings ---")
print(loadings)
print(f"\nTotal Variance Explained: {sum(vaf):.2f}%")

# Plotting
fig = plt.figure(figsize=(16, 7))

# Subplot 1: 3D Scatter (Corrected Axis Logic)
ax1 = fig.add_subplot(121, projection='3d')
for species in y.unique():
    mask = y == species
    ax1.scatter(components[mask, 0], components[mask, 1], components[mask, 2], label=species)

ax1.set_xlabel(f'PC1 ({vaf[0]:.1f}%)')
ax1.set_ylabel(f'PC2 ({vaf[1]:.1f}%)')
ax1.set_zlabel(f'PC3 ({vaf[2]:.1f}%)') # Fixed from PC2 to PC3
ax1.set_title("3D PCA Projection")

# Subplot 2: 2D Scatter
ax2 = fig.add_subplot(122)
for species in y.unique():
    mask = y == species
    ax2.scatter(components[mask, 0], components[mask, 1], label=species)
ax2.set_xlabel(f'PC1 ({vaf[0]:.1f}%)')
ax2.set_ylabel(f'PC2 ({vaf[1]:.1f}%)')
ax2.set_title("2D PCA Projection")

plt.legend()
plt.tight_layout()
plt.show()
