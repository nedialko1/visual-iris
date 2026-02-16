import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


class MNISTLinearDiscriminant:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(n_components=9)
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # Fetching MNIST 784 (28x28 pixels)
        print("Fetching MNIST data...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target

        # Split and Scale
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_lda(self):
        print("Training LDA and projecting to 9 dimensions...")
        self.lda.fit(self.X_train, self.y_train)

    def get_equations(self):
        # Coefficients (Weights): 10 classes x 784 features = 7840
        # Intercepts (Bias): 10 classes = 10
        weights = self.lda.coef_
        biases = self.lda.intercept_
        total_params = weights.size + biases.size

        return {
            "weights_shape": weights.shape,
            "bias_shape": biases.shape,
            "total_parameters": total_params
        }

    def evaluate(self):
        y_pred = self.lda.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\nLDA Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        return acc


# Execution
model = MNISTLinearDiscriminant()
model.load_data()
model.train_lda()
equations = model.get_equations()
accuracy = model.evaluate()

print(f"Total Parameters in Classification Equations: {equations['total_parameters']}")