## === netron_t00.py == netron visualizer primer
import netron
import torch.onnx

# Load or define your model

# Define the output file name
onnx_file_path = "iris.onnx"

# Typical Iris Architecture for Netron

import torch
import torch.nn as nn

sizes = [4, 8, 9, 3]

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(sizes[0], sizes[1])  # 4 inputs (features) -> 8 neurons
        self.fc2 = nn.Linear(sizes[1], sizes[2]) # 8 -> 9 hidden neurons
        self.fc3 = nn.Linear(sizes[2], sizes[3])  # 9 -> 3 output classes (species)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = IrisNet()

# Export for Netron
dummy_input = torch.randn(1, 4)
torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True)
print(f"Model exported to {onnx_file_path}")

# ===================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, title="Hinton Diagram", max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()


    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())


    for (x, y), w in np.ndenumerate(matrix):
        if abs(w) < max_weight/4:
            color = 'white' if w > 0 else 'black'
        else:
            color = 'green' if w > 0 else 'red'

        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)


    ax.autoscale_view()
    ax.invert_yaxis()
    plt.title(title)
    plt.ylabel("Target Neurons (Output)")
    plt.xlabel("Source Neurons (Input)")


# --- Example Usage with IrisNet ---
# Assuming the IrisNet model class from above:
# model = IrisNet()
# Note: Transpose the PyTorch weights which are [in_features, out_features]
weights = model.fc1.weight # This is a [8, 4] tensor

plt.figure(figsize=(10, 8))
hinton(model.fc1.weight.T, title=f"IrisNet: FC1 Weights ({sizes[0]} Inputs -> {sizes[1]} Neurons)")

plt.figure(figsize=(10, 8))
hinton(model.fc2.weight.T, title=f"IrisNet: FC2 Weights ({sizes[1]} Inputs -> {sizes[2]} Neurons)")
plt.show()

"""

# 1. Start the server
netron.start(onnx_file_path, address=('localhost', 8080))

# 2. Keep the script running so the server doesn't close
input("Press Enter to stop the Netron server...")

"""

