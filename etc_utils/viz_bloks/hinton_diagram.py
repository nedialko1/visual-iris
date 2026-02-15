import torch

import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, title="Hinton Diagram", max_weight=None, ax=None):
# Draws Hinton diagrams visualizing a weight matrix or model drift
    ax = ax if ax is not None else plt.gca()

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    # ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())


    for (y, x), w in np.ndenumerate(matrix):
        if abs(w) < max_weight/4:
            color = 'white' if w > 0 else 'black'
        else:
            color = 'green' if w > 0 else 'red'

        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    # ax.invert_yaxis()
    plt.title(title)
    plt.ylabel("Output (Targets)")
    plt.xlabel("Input")

"""
# --- Example Usage with IrisNet ---
model = IrisNet()
weights = model.fc1.weight 

plt.figure(figsize=(10, 8))
# transpose because PyTorch weights are [in_features, out_features]
hinton(model.fc1.weight.T)

plt.show()
"""
