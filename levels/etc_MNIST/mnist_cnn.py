import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Seed 42 for reproducibility
from seed_models import set_seed
# set_seed(42)

class ParametricMicroCNN(nn.Module):
    def __init__(self, n_filters=28):
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


def train_and_monitor(n_filters=28, epochs=7):
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

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParametricMicroCNN(n_filters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"CNN Specs: {n_filters} filters | {param_count} parameters")
    print(f"LDA Baseline Reference: ~87.5% (7,850 params)")
    print("-" * 50)

    # 3. Initial Accuracy (Random Weights)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            correct += (output.argmax(1) == target.to(device)).sum().item()
    print(f"Initial (Epoch 0) Accuracy: {100 * correct / len(y_test):.2f}%")

    # 4. Training Rounds
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

        # Post-Round Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                correct += (output.argmax(1) == target.to(device)).sum().item()

        current_acc = 100 * correct / len(y_test)
        print(f"End of Round {epoch + 1} | Accuracy: {current_acc:.2f}%")

        if current_acc > 87.5:
            print(f" >>> NOTE: CNN has surpassed LDA baseline.")

            return model, test_loader

# Execute
final_model, test_loader = train_and_monitor(n_filters=28)  #  5

import matplotlib.pyplot as plt

# --------------------------------

from pathlib import Path

# Get the path of the current script file
script_path = Path(__file__).resolve()
# Get the directory containing *this* script
script_dir = script_path.parent

# --------------------------------

def visualize_deviants(model, loader, target_class, pred_class, num_samples=5):
    # Detect the model's current home
    device = next(model.parameters()).device
    model.eval()
    samples = []

    with torch.no_grad():
        for data, target in loader:
            # FIX: Move input data to the same device as the model weights
            data_dev = data.to(device)

            output = model(data_dev)
            preds = output.argmax(dim=1)

            # Find indices where Actual is target_class but Predicted is pred_class
            mask = (target == target_class) & (preds.cpu() == target_class)  # Keep logic on CPU

            # Re-evaluating the mask on CPU for array indexing
            preds_cpu = preds.cpu().numpy()
            target_cpu = target.numpy()

            error_indices = np.where((target_cpu == target_class) & (preds_cpu == pred_class))[0]

            for idx in error_indices:
                samples.append(data[idx][0].numpy())  # Keep original CPU image for plotting
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

    # Plotting logic
    if not samples:
        print(f"No samples found for {target_class} -> {pred_class}")
        return

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(f"Forensic Evidence: Actual {target_class} classified as {pred_class}", fontsize=16)
    for i, img in enumerate(samples):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    if target_class == 4:
        save_path = f"../../images/Appendix_1/MNIST_CNN_{target_class}_as_{pred_class}.png"
        save_path = script_dir / Path(save_path)
        plt.savefig(save_path)

    plt.show()

# --- RUNNING THE AUDIT ---
# 1. The 'Black Hole' Cluster (5 classified as 3)
visualize_deviants(final_model, test_loader, target_class=5, pred_class=3)

# 2. The 'Balanced' Cluster (4 classified as 9)
visualize_deviants(final_model, test_loader, target_class=4, pred_class=9)


def forensic_visualizer(model, loader, target_class=5, pred_class=3):
    device = next(model.parameters()).device
    model.eval()

    # 1. FIND THE 'UGLY' SAMPLE
    found_img = None
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Mask for the specific error (e.g., Actual 5, Predicted 3)
            mask = (labels.to(device) == target_class) & (preds == pred_class)
            if mask.any():
                found_img = images[mask][0]
                break

    if found_img is None:
        print(f"No samples found for error: {target_class} -> {pred_class}")
        return

    # 2. DYNAMICALLY FIND THE CONV LAYER
    # We look for the first instance of nn.Conv2d within the model
    first_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_conv = module
            break

    if first_conv is None:
        print("Could not find a Convolutional layer in this model.")
        return

    # 3. EXTRACT ACTIVATIONS
    with torch.no_grad():
        # Pass the image through that specific layer
        activations = first_conv(found_img.unsqueeze(0))
        # Apply ReLU for better visualization (simulating the model's flow)
        activations = torch.relu(activations)

    # 4. DISPLAY THE INTERNAL REALITY
    act_map = activations.squeeze(0).cpu().numpy()
    n_f = act_map.shape[0]

    cols = 7
    rows = (n_f // cols) + (1 if n_f % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    fig.suptitle(f"CNN Forensic: First Layer Reactions to an 'Ugly {target_class}'", fontsize=16)

    for i in range(rows * cols):
        ax = axes.flatten()[i]
        if i < n_f:
            ax.imshow(act_map[i], cmap='magma')
            ax.set_title(f"Filter {i}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- EXECUTION ---
# This will work regardless of whether your attribute is .features or .conv_block
forensic_visualizer(final_model, test_loader)