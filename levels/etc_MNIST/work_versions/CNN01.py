import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class MNISTFactory:
    def __init__(self, n_filters=24):
        self.n_filters = n_filters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define Architecture
        self.model = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),  # Global Average Pool
            nn.Flatten(),
            nn.Linear(n_filters, 10)
        ).to(self.device)

        self.params = sum(p.numel() for p in self.model.parameters())

    def train_model(self, train_loader, epochs=3):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(data), target)
                loss.backward()
                optimizer.step()
        print(f"Trained CNN with {self.n_filters} filters ({self.params} params).")

    def get_deviants(self, test_loader):
        self.model.eval()
        misclassified = []
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data.to(self.device))
                preds = output.argmax(dim=1).cpu().numpy()
                targets = target.numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

                # Find index of errors in this batch
                errors = np.where(preds != targets)[0]
                for idx in errors:
                    if len(misclassified) < 100:  # Limit for memory
                        misclassified.append((data[idx][0], targets[idx], preds[idx]))

        return all_targets, all_preds, misclassified


# --- EXECUTION ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000)

# Target: 28 filters (~7,950 params) to match LDA budget
factory = MNISTFactory(n_filters=28)
factory.train_model(train_loader)
y_true, y_pred, deviants = factory.get_deviants(test_loader)