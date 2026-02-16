import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

N_COMPONENTS = 20    # 5 or 10 or [20] or 28 or [40] or 100 or 328 ?

# Seed 42 for reproducibility
from seed_models import set_seed
set_seed(42)

class ParametricMicroCNN(nn.Module):
    def __init__(self, n_filters=N_COMPONENTS):
        super().__init__()
        self.n_filters = n_filters
        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )
        self.classifier = nn.Linear(n_filters, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD & SCALE
with np.load("mnist_data.npz") as d:
    X, y = d['X'], d['y']

# Standardization is critical for PCA to treat all pixel variances fairly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=10000, random_state=42
)

# 2. PCA DE-NOISING (Capturing 95% of variance)
# Usually around 150-200 components for MNIST?
pca = PCA(n_components=N_COMPONENTS)  # n_components=0.95, svd_solver='full'
X_train_pca = pca.fit_transform(X_train)

# Access the attributes
retained_count = pca.n_components_
variance_ratios = pca.explained_variance_ratio_
total_variance = variance_ratios.sum()

print(f"PCA retained {pca.n_components} components - they explain {100*total_variance:.3f}% of the variance.")

X_test_pca = pca.transform(X_test)

# 3. LDA CLASSIFICATION
lda_refined = LinearDiscriminantAnalysis()
lda_refined.fit(X_train_pca, y_train)

# --- EXECUTION ---
fresh_cnn = ParametricMicroCNN(n_filters=N_COMPONENTS)

def surgical_initialization_v2(model, lda_model, pca_model):
    model.eval()

    # 1. THE EYES: Reconstruct 784-px ghosts from PCA space
    weights_784 = lda_model.coef_ @ pca_model.components_
    ghosts = weights_784.reshape(10, 28, 28)

    # Locate the layers dynamically
    conv_layer = None
    fc_layer = None

    for module in model.modules():
        if isinstance(module, nn.Conv2d) and conv_layer is None:
            conv_layer = module
        if isinstance(module, nn.Linear):
            fc_layer = module  # We'll take the last one found

    with torch.no_grad():
        # Transplant the 'heart' of the LDA ghosts into CNN kernels
        if conv_layer is not None:
            for i in range(10):
                # Extract the central 3x3 'discriminant' patch
                patch = ghosts[i, 12:15, 12:15]
                # Normalize to maintain signal stability
                patch = (patch - patch.mean()) / (patch.std() + 1e-5)
                conv_layer.weight[i, 0] = torch.from_numpy(patch).float()

        # 2. THE BRAIN: Identity Mapping (Filter i -> Class i)
        if fc_layer is not None:
            nn.init.zeros_(fc_layer.weight)
            # Create a direct 'Highway' for the first 10 filters
            for i in range(10):
                if i < fc_layer.weight.shape[1] and i < fc_layer.weight.shape[0]:
                    fc_layer.weight[i, i] = 1.0
            if fc_layer.bias is not None:
                nn.init.zeros_(fc_layer.bias)

    print("Transplant V2 Complete: Generic Layer Access Successful.")
    return model

# 1. MATCH BANDWIDTH: PCA with 28 Components
pca_28 = PCA(n_components=N_COMPONENTS)
X_train_pca_28 = pca_28.fit_transform(X_train)

# 2. LDA RE-FIT
lda_28 = LinearDiscriminantAnalysis()
lda_28.fit(X_train_pca_28, y_train)

# 3. EXTRACT BALANCED WEIGHTS
# Now weights_28 has the same dimensionality as our filter count
weights_28 = lda_28.coef_ @ pca_28.components_

# --------------------------------

from pathlib import Path

# Get the path of the current script file
script_path = Path(__file__).resolve()
# Get the directory containing *this* script
script_dir = script_path.parent

# --------------------------------

from scipy.ndimage import gaussian_filter

def visualize_lda_weights(weights):
    # Calculate the 'Global Average Ghost'
    mean_weight = np.mean(weights, axis=0)

    fig = plt.figure(figsize=(16, 8), facecolor='white')
    fig.tight_layout()
    gs = fig.add_gridspec(2, 6,
      width_ratios=[1, 1, 1, 1, 1, 0.2], left=0.1, top=0.9, wspace=0.3, hspace=0.1)

    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    plt.subplots_adjust(top=0.8)
    """

    fig.suptitle("LDA Templates: What makes a digit UNIQUE?", fontsize=14, y=0.98)
    txt = f"[Based on {pca_28.n_components} PCs explaining {100*total_variance:.3f}% of variance]"
    plt.title(txt, y=1.05)
    plt.axis('off')

    for i in range(10):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        # ax = axes[i // 5, i % 5]

        # SUBTRACT the mean to see the unique 'Discriminant' features
        contrast_weight = (weights[i] - mean_weight).reshape(28, 28)

        # Blur slightly to remove pixel noise
        # contrast_weight = gaussian_filter(contrast_weight, sigma=0.8)

        vbound = np.max(np.abs(contrast_weight)) * 0.8
        norm = colors.TwoSlopeNorm(vmin=-vbound, vcenter=0, vmax=vbound)

        im = ax.imshow(contrast_weight, cmap='RdBu_r', norm=norm)
        ax.set_title(f"Signature ({i})")
        ax.axis('off')

    # single axes for the colorbar that spans BOTH rows in the last column
    cax = fig.add_subplot(gs[:, 5])
    plt.colorbar(im, cax, shrink=0.5)
    # plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)

    save_path = "../../images/Appendix_1/MNIST_LDA_2D_weights.png"
    save_path = script_dir / Path(save_path)

    plt.savefig(save_path)
    plt.show()

visualize_lda_weights(weights_28)

# 4. SURGERY: The Symmetric Initialization
# We use our self-correcting bridge to inject these 28-PC skeletons
frankenstein_cnn = surgical_initialization_v2(fresh_cnn, lda_28, pca_28)

# --- EXECUTION ---
# frankenstein_cnn = surgical_initialization_v2(fresh_cnn, lda_refined, pca)

def get_test_loader():
    # 1. Load from the .npz archive established in previous turn
    with np.load("mnist_data.npz") as data:
        X, y = data['X'], data['y']

    # Reshape for CNN: (N, 1, 28, 28)
    X = X.reshape(-1, 1, 28, 28)

    # Split
    X_train, X_test = torch.FloatTensor(X[:60000]), torch.FloatTensor(X[60000:])
    y_train, y_test = torch.LongTensor(y[:60000]), torch.LongTensor(y[60000:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1000)

    return test_loader

test_loader = get_test_loader()

# Quick Test Loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = frankenstein_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Zero-Training 'Frankenstein' Accuracy: {100 * correct / total:.2f}%")

# -------------------------------------------

# 1. Setup Optimizer
optimizer = optim.Adam(frankenstein_cnn.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
frankenstein_cnn.train()

# 2. Train for just 10 small batches
print("Attempting to 'Wake Up' the Frankenstein brain...")
for i, (images, labels) in enumerate(test_loader):
    if i >= 10: break  # Only 10 batches!

    optimizer.zero_grad()
    outputs = frankenstein_cnn(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 3. Re-test Accuracy
frankenstein_cnn.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = frankenstein_cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Post-Wakeup Accuracy (After 10 batches): {100 * correct / total:.2f}%")