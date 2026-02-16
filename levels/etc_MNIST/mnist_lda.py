from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Seed 42 for reproducibility
from seed_models import set_seed
set_seed(42)

def execute_lda_benchmark(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

    # LDA uses (C-1) components = 9 dimensions
    lda = LinearDiscriminantAnalysis(n_components=9)
    # lda.fit(X_train, y_train)
    lda.fit(X, y)

    # Parameter check: (784 features * 10 classes) + 10 biases
    total_params = lda.coef_.size + lda.intercept_.size
    accuracy = lda.score(X_test, y_test)

    print(f"LDA Parameters: {total_params}")
    print(f"LDA Accuracy:   {accuracy:.2%}")

    y_pred = lda.predict(X_test)
    return lda, confusion_matrix(y_test, y_pred), X_test, y_test, y_pred, X_train, y_train

from get_mnist import sync_mnist_numpy

# Load baseline data
X, y = sync_mnist_numpy()

import numpy as np

def binarize_and_benchmark(X, y):
    # Threshold at 0.5: Ink is 1, Paper is 0
    X_bin = (X > 0.5).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=10000, random_state=42)

    # Run LDA
    lda_bin = LinearDiscriminantAnalysis(n_components=9)
    lda_bin.fit(X_train, y_train)

    acc = lda_bin.score(X_test, y_test)
    print(f"Binarized LDA Accuracy: {acc:.2%}")

    y_pred = lda_bin.predict(X_test)

    return lda_bin, confusion_matrix(y_test, y_pred), X_test, y_test, y_pred, X_train, y_train

#
lda_model, lda_cm, X_test, y_test, y_pred_lda, X_train, y_train = execute_lda_benchmark(X, y)

# lda_model, lda_cm, X_test, y_test, y_pred_lda, X_train, y_train = binarize_and_benchmark(X, y)

import matplotlib.pyplot as plt

# --------------------------------

from pathlib import Path

# Get the path of the current script file
script_path = Path(__file__).resolve()
# Get the directory containing *this* script
script_dir = script_path.parent

# --------------------------------

def visualize_lda_deviants(lda_model, X_test, y_test, target_class, pred_class, num_samples=5):
    """
    X_test should be the flattened (N, 784) arrays used for LDA.
    """
    # Get predictions
    y_pred = lda_model.predict(X_test)

    # Identify indices where the error occurs
    error_indices = np.where((y_test == target_class) & (y_pred == pred_class))[0]

    if len(error_indices) == 0:
        print(f"No samples found for LDA error: {target_class} -> {pred_class}")
        return

    # Plotting
    fig, axes = plt.subplots(1, min(num_samples, len(error_indices)), figsize=(15, 3))
    fig.suptitle(f"LDA Forensic Evidence: Actual {target_class} classified as {pred_class}", fontsize=16)

    for i in range(min(num_samples, len(error_indices))):
        idx = error_indices[i]
        img = X_test[idx].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Index: {idx}")
        axes[i].axis('off')

    save_path = f"../../images/Appendix_1/MNIST_LDA_{target_class}_as_{pred_class}.png"
    save_path = script_dir / Path(save_path)
    plt.savefig(save_path)
    plt.show()


# --- RUNNING THE LDA AUDIT ---
# Using the objects from our previous LDA execution
# visualize_lda_deviants(lda_model, X_test, y_test, target_class=5, pred_class=3)

visualize_lda_deviants(lda_model, X_test, y_test, target_class=4, pred_class=9)

import matplotlib.colors as colors

from scipy.ndimage import gaussian_filter

def visualize_lda_refined(lda_model):
    # Retrieve the weights (10, 784)
    weights = lda_model.coef_

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("The 10 Linear Equations: Refined Spatial Templates", fontsize=16)

    # Calculate global min/max for a consistent scale across all 10 digits
    # Use a percentile to prevent single noisy pixels from ruining the scale
    vbound = np.percentile(np.abs(weights), 98)
    norm = colors.TwoSlopeNorm(vmin=-vbound, vcenter=0, vmax=vbound)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        # Reshape and apply a light blur (sigma=0.5) to clear the 'static'
        digit_weight = weights[i].reshape(28, 28)
        digit_weight_smooth = gaussian_filter(digit_weight, sigma=0.5)

        # RdBu_r: Red = Positive (WANT), Blue = Negative (FORBIDDEN)
        im = ax.imshow(digit_weight_smooth, cmap='RdBu_r', norm=norm)
        ax.set_title(f"Template for '{i}'")
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.show()


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. ARCHITECTURAL UPGRADE: PCA + LDA Pipeline
# We take 50 components to capture the 'skeleton' without the 'blob'
forensic_pipe = Pipeline([
    ('pca', PCA(n_components=50)),
    ('lda', LinearDiscriminantAnalysis())
])

forensic_pipe.fit(X_train, y_train)

# 2. EXTRACTING THE 'DE-NOISED' WEIGHTS
# We project the LDA coefficients back from PCA-space to Image-space
# Weights = (LDA_coef @ PCA_components)
pca_comp = forensic_pipe.named_steps['pca'].components_
lda_coef = forensic_pipe.named_steps['lda'].coef_
denoised_weights = lda_coef @ pca_comp

def visualize_denoised_ghosts(weights):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("De-Noised LDA Ghosts: PCA-Filtered Skeletons", fontsize=16)

    v = np.percentile(np.abs(weights), 98)
    norm = colors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        w_img = weights[i].reshape(28, 28)
        # Apply a subtle filter to highlight the structure
        im = ax.imshow(w_img, cmap='RdBu_r', norm=norm)
        ax.set_title(f"Cleaned '{i}'")
        ax.axis('off')
    plt.show()

visualize_denoised_ghosts(denoised_weights)

def visualize_lda_contrast(lda_model):
    weights = lda_model.coef_
    # Calculate the 'Global Average Ghost'
    mean_weight = np.mean(weights, axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("LDA Contrast Templates: What makes each digit UNIQUE?", fontsize=16)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        # SUBTRACT the mean to see the unique 'Discriminant' features
        contrast_weight = (weights[i] - mean_weight).reshape(28, 28)

        # Blur slightly to remove pixel noise
        contrast_weight = gaussian_filter(contrast_weight, sigma=0.8)

        vbound = np.max(np.abs(contrast_weight)) * 0.8
        norm = colors.TwoSlopeNorm(vmin=-vbound, vcenter=0, vmax=vbound)

        im = ax.imshow(contrast_weight, cmap='RdBu_r', norm=norm)
        ax.set_title(f"Unique Signature: {i}")
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    plt.show()

# --- EXECUTION ---
# visualize_lda_contrast(lda_model)
## visualize_lda_refined(lda_model)