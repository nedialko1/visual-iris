import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Seed 42 for reproducibility
from seed_models import set_seed
set_seed(42)

# --- 1. MODEL FACTORY ---
class FinalMicroCNN(nn.Module):
    def __init__(self, n_filters=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(n_filters, 10)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# --- 2. DATA & TRAINING ---
with np.load("mnist_data.npz") as d:
    X, y = d['X'].reshape(-1, 1, 28, 28), d['y']

X_train, X_test = torch.FloatTensor(X[:60000]), torch.FloatTensor(X[60000:])
y_train, y_test = torch.LongTensor(y[:60000]), torch.LongTensor(y[60000:])
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1000)

model = FinalMicroCNN(n_filters=28).to('cpu')  # Running on CPU for stability
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# Log initial state
print(f"Total Parameters: {sum(p.numel() for p in model.parameters())} (Matching LDA budget)")

nEpochs = 2

# Training Loop
for epoch in range(nEpochs):

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for b_x, b_y in test_loader:
            correct += (model(b_x).argmax(1) == b_y).sum().item()

    print(f"Round {epoch} Accuracy: {100 * correct / 10000:.2f}%")

    model.train()
    for b_x, b_y in loader:
        optimizer.zero_grad()
        criterion(model(b_x), b_y).backward()
        optimizer.step()

    if epoch == nEpochs-1:
        # Re-Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for b_x, b_y in test_loader:
                correct += (model(b_x).argmax(1) == b_y).sum().item()

        print(f"Round {epoch+1} Accuracy: {100 * correct / 10000:.2f}%")

# --- 3. CONFUSION MATRIX & DEVIANT EXTRACTION ---
all_preds = []
with torch.no_grad():
    for b_x, _ in test_loader:
        all_preds.extend(model(b_x).argmax(1).numpy())

cm = confusion_matrix(y_test, all_preds)

print(cm)