import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD & SCALE
with np.load("../mnist_data.npz") as d:
    X, y = d['X'], d['y']

# Standardization is critical for PCA to treat all pixel variances fairly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=10000, random_state=42
)

# 2. PCA DE-NOISING (Capturing 95% of variance)
# Usually around 150-200 components for MNIST
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"PCA retained {pca.n_components_} components.")

# 3. LDA CLASSIFICATION
lda_refined = LinearDiscriminantAnalysis()
lda_refined.fit(X_train_pca, y_train)

acc = lda_refined.score(X_test_pca, y_test)
print(f"Final PCA-LDA Accuracy: {acc:.2%}")

# 4. BACK-PROJECTION FOR VISUALIZATION
# Project LDA weights (10, n_components) back to (10, 784)
# Weights_original_space = LDA_coef @ PCA_components
lda_weights_784 = lda_refined.coef_ @ pca.components_

# ------------------------------

def visualize_pca_lda_weights(weights):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("PCA-LDA Refined Weight Space: The Discriminant Skeletons", fontsize=16)

    # Use a robust scaler for the visualization colors
    vbound = np.percentile(np.abs(weights), 99)
    norm = colors.TwoSlopeNorm(vmin=-vbound, vcenter=0, vmax=vbound)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        w_img = weights[i].reshape(28, 28)

        # Plotting the 'Cleaned' template
        im = ax.imshow(w_img, cmap='RdBu_r', norm=norm)
        ax.set_title(f"Template '{i}'")
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.show()


visualize_pca_lda_weights(lda_weights_784)