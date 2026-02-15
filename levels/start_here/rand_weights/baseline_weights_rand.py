import numpy as np

# Set seed for reproducible "imperfect" results
# np.random.seed(42)
seed = int(np.random.rand() * (2**32 - 1))
print(f"********* seed = {seed}")
np.random.seed(seed)

def generate_doctored_weights():
    # 4 -> 8
    W1 = np.random.randn(4, 8) * 0.6
    b1 = np.random.randn(1, 8) * 0.1

    # 8 -> 9
    W2 = np.random.randn(8, 9) * 0.5
    b2 = np.random.randn(1, 9) * 0.1

    # 9 -> 3
    # We slightly scale down the final layer to prevent over-confidence
    W3 = np.random.randn(9, 3) * 0.4
    b3 = np.random.randn(1, 3) * 0.05

    return [W1, b1, W2, b2, W3, b3]


weights = generate_doctored_weights()
