import numpy as np
import os
from sklearn.datasets import fetch_openml


def sync_mnist_numpy(filename="mnist_data.npz"):
    if not os.path.exists(filename):
        print("Fetching MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        # Standardize: Pixels [0, 255] -> [0.0, 1.0]
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int8)

        np.savez_compressed(filename, X=X, y=y)
        print(f"Data archived to {filename}")
    else:
        print(f"Loading from local archive: {filename}")

    with np.load(filename) as data:
        return data['X'], data['y']


# Load baseline data
# X, y = sync_mnist_numpy()