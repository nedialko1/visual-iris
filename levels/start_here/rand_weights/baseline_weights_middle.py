import numpy as np

np.random.seed(42)


def get_averaged_weights():
    # Helper to create the 3 archetypes
    def get_v1():  # Virginica Dominant
        return [np.random.randn(4, 8) * 0.6, np.zeros((1, 8)), np.random.randn(8, 9) * 0.5,
                np.zeros((1, 9)), np.random.randn(9, 3) * 0.4, np.array([[-0.5, -0.5, 2.0]])]

    def get_v2():  # High Entropy (Uniform Noise)
        return [np.random.normal(0, 0.1, (4, 8)), np.zeros((1, 8)), np.random.normal(0, 0.1, (8, 9)),
                np.zeros((1, 9)), np.random.normal(0, 0.05, (9, 3)), np.zeros((1, 3))]

    def get_v3():  # Setosa-Breaker (Versicolor Bias)
        return [np.random.normal(0, 0.1, (4, 8)), np.zeros((1, 8)), np.random.normal(0, 0.1, (8, 9)),
                np.zeros((1, 9)), np.random.normal(0, 0.05, (9, 3)), np.array([[-2.5, 1.2, 1.1]])]

    # Generate and average
    models = [get_v1(), get_v2(), get_v3()]
    avg_weights = []

    for i in range(6):  # Loop through W1, b1, W2, b2, W3, b3
        layer_avg = np.mean([m[i] for m in models], axis=0)
        avg_weights.append(layer_avg)

    return avg_weights


weights = get_averaged_weights()
