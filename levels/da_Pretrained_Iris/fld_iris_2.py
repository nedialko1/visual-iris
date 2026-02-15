import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Seed 42 for reproducibility
seed = 42
np.random.seed(seed)

# --- Setup Data ---
iris = load_iris()
X, y = iris.data, iris.target
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# --- Compute Fisher Planes ---
def get_plane(X_p, y_p, c1, c2):
    m1, m2 = np.mean(X_p[np.isin(y_p, c1)], axis=0), np.mean(X_p[np.isin(y_p, c2)], axis=0)
    sw = np.dot((X_p[np.isin(y_p, c1)]-m1).T, (X_p[np.isin(y_p, c1)]-m1)) + \
         np.dot((X_p[np.isin(y_p, c2)]-m2).T, (X_p[np.isin(y_p, c2)]-m2))
    w = np.linalg.solve(sw, (m1 - m2))
    return w, -0.5 * np.dot(w, (m1 + m2))

w_set, b_set = get_plane(X_pca, y, [0], [1, 2]) # Setosa vs Others
w_vv, b_vv   = get_plane(X_pca, y, [1], [2])    # Versicolor vs Virginica

# --- The Contraption ---
class IrisLogicNet(nn.Module):
    def __init__(self, pca_comp, w_s, b_s, w_v, b_v):
        super().__init__()

        # H1: PCA Projection (Fixed)
        self.h1 = nn.Linear(4, 3)
        self.h1.weight.data = torch.tensor(pca_comp, dtype=torch.float32)
        self.h1.bias.data.fill_(0)

        # H2: Fisher ReLUs (Fixed)
        self.h2 = nn.Linear(3, 3)
        self.h2.weight.data = torch.tensor(np.array([w_s, w_v, -w_v]), dtype=torch.float32)
        self.h2.bias.data = torch.tensor(np.array([[b_s, b_v, -b_v]]), dtype=torch.float32)

        # Output: Your Manual Logic
        self.out = nn.Linear(3, 3)

        with torch.no_grad():
            # Unit 1 (Setosa) -> Out 1 (Big), kills others
            self.out.weight[:,0] = torch.tensor([10.0, -5.0, -5.0], dtype=torch.float32)
            # Unit 2 (Versi)  -> Out 2 (Drive), inhibits Out 1 (Wee bit)
            self.out.weight[:,1] = torch.tensor([-5, 5.0, -2.0], dtype=torch.float32)
            # Unit 3 (Virgi)  -> Out 3 (Drive)
            self.out.weight[:,2] = torch.tensor([-2.0, -2.0, 5.0], dtype=torch.float32)

            self.out.bias.data.fill_(0)
            # self.out.bias.data = torch.tensor(np.array([[-5, -1, -1]]), dtype=torch.float32)

    def forward(self, x):
        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        return self.out(x)

# --- Test Run ---
model = IrisLogicNet(pca.components_, w_set, b_set, w_vv, b_vv)

model.load_state_dict(torch.load('run_logs/pca3_FLD_3.pt'))

print(f"W3 = \n {model.out.weight.data.numpy()}")
print(f"b3 = {model.out.bias.data.numpy()}")

# ==============================

# Create a .py output with the model's weights:

txt = f"""
# >>> import levels.da_Pretrained_Iris.Refined_2 as R2W

import numpy as np

# --- (fc1.weight) ([4, 3]) ---
W1 = np.array(\\
{np.array2string(model.h1.weight.data.numpy(), separator=', ')}
)

# --- (fc1.bias) ([3]) ---
b1 = np.array(\\
{np.array2string(model.h1.bias.data.numpy(), separator=', ')}
)

# --- (fc2.weight) ([3, 3]) ---
W2 = np.array(\\
{np.array2string(model.h2.weight.data.numpy(), separator=', ')}
)

# --- (fc2.bias) ([3]) ---
b2 = np.array(\\
{np.array2string(model.h2.bias.data.numpy().flatten(), separator=', ')}
)

# --- (fc3.weight) ([3, 3]) ---
W3 = np.array(\\
{np.array2string(model.out.weight.data.numpy(), separator=', ')}
)

# --- (fc3.bias) ([3]) ---
b3 = np.array(\\
{np.array2string(model.out.bias.data.numpy(), separator=', ')}
)
"""
with open("Refined_2.py", "w") as file:
    file.write(txt)