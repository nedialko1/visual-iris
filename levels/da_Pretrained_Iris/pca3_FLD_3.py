import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Seed 42 for reproducibility
seed = 42
np.random.seed(seed)

# Function to create a surface for the plane Ax + By + Cz + D = 0
# Simplified to z = -(Ax + By + D)/C
def plot_plane(ax, w, b, color, alpha=0.3):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    # w[0]x + w[1]y + w[2]z + b = 0  => z = (-w[0]x - w[1]y - b) / w[2]
    zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, shade=False)

# --- Compute Fisher Planes ---
def get_plane(X_p, y_p, c1, c2):
    m1, m2 = np.mean(X_p[np.isin(y_p, c1)], axis=0), np.mean(X_p[np.isin(y_p, c2)], axis=0)
    sw = np.dot((X_p[np.isin(y_p, c1)]-m1).T, (X_p[np.isin(y_p, c1)]-m1)) + \
         np.dot((X_p[np.isin(y_p, c2)]-m2).T, (X_p[np.isin(y_p, c2)]-m2))
    w = np.linalg.solve(sw, (m1 - m2))
    return w, -0.5 * np.dot(w, (m1 + m2))

from sklearn.preprocessing import StandardScaler

# --- Setup Data ---
iris = load_iris()
X, y = iris.data, iris.target
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

w_set, b_set = get_plane(X_pca, y, [0], [1, 2]) # Setosa vs Others
w_vv, b_vv   = get_plane(X_pca, y, [1], [2])    # Versicolor vs Virginica

from etc_utils.viz_bloks import smart_show

import matplotlib.pyplot as plt

plt.figure(figsize=(11, 10))
# Subplot 1: Scatter with variety
plt.subplot(1, 1, 1, projection='3d')
ax = plt.gca()

x1=X_pca[:, 0]
y2=X_pca[:, 1]
z3=X_pca[:, 2]

ax.set_xlim(np.min(x1), np.max(x1))
ax.set_ylim(np.min(y2), np.max(y2))

# Using the w, b values from our previous successful run
# (Assuming w_set, b_set and w_vv, b_vv are defined)
# plot_plane(ax, w_set, b_set, 'cyan')   # Setosa Boundary
plot_plane(ax, w_vv, b_vv, 'gray')  # V-V Boundary

species = ['Setosa', 'Versicolor', 'Virginica']
colors=['r', 'g', 'b']
styles=['o', 'x', 's']

yy=np.unique(y)

for i, y_class in enumerate(yy):
    # Filter your data for just this category
    mask = (y == y_class)
    ax.scatter(x1[mask], y2[mask], z3[mask], s=100,
        color=colors[i], marker=styles[i], label=species[i])

ax.legend()
ax.view_init(elev=25, azim=-60, roll=0)

plt.title('3D PCA Projection: Variety Clusters')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')

# plt.show()
smart_show.smart_show(fgSaveFigures=True,
                      selectedFigures={1: "pca_fld_3D"})

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

# Convert y labels to tensors long
y_true = torch.LongTensor(y)
X = torch.tensor(X_std, dtype=torch.float32)

out = model.forward(X)
print(f'*** [out]: {out.shape}')
preds = torch.argmax(out, dim=1).numpy()

n_total = len(y)

print(f"*** {preds.shape} <> {n_total}")

idx_ok = np.where(preds == y)[0]
n_ok = len(idx_ok)
idx_err = np.where(preds != y)[0]
n_err = len(idx_err)

print(f"??? {preds[idx_err]} ")
print(f"^^^ {y[idx_err]} ")

accuracy = n_ok / n_total
print(f"Accuracy : {n_ok} / {n_total} -> {accuracy * 100:.2f}%")

accuracy = np.mean(preds == y)
print(f"Zero-Training Accuracy: {accuracy * 100:.2f}%")

# ==============================
# Set the criterion of model to measure the error, how far off the predictions are from the data
FG_MSE_LOSS = False

if FG_MSE_LOSS:
    criterion = nn.MSELoss()
else:
    criterion =  nn.CrossEntropyLoss()

if FG_MSE_LOSS:
    # ------------------------------
    ns = 3
    y_value =  np.zeros((n_total,3))

    """
    """

    for iTest in  range(ns):
      iiThisClass = np.where(y == iTest)[0]
      print(f"*** iiThisClass.shape = {iiThisClass.shape}")
      yThis = np.zeros((1,3))
      yThis[0,iTest] = 1
      N_REPLICATE = len(iiThisClass)
      y_value[iiThisClass] = np.tile(yThis, (N_REPLICATE, 1))

    y_test = torch.tensor(y_value, dtype=torch.float32)

    # Validate y_test assignments for 5 samples of each class:
    idx = [ i for i in range(5)  ]
    for iTest in  range(ns):
        iiThisClass = np.where(y == iTest)[0]
        print(f"{iTest}. <> \n {y_value[iiThisClass[idx]]} ")

# -----------------------------
# Train our model!

target_layer_prefix = "out."

for name, param in model.named_parameters():
    if target_layer_prefix in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# Epochs? (one run through all the training data in our network)
epochs = 900
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X) # Get predicted results

  # print(f"*** {y_pred.shape} <> {y_test.shape}")

  # Measure the loss/error, initially high: predicted values vs training data
  if FG_MSE_LOSS:
    loss = criterion(y_pred, y_test)
  else:
    loss = criterion(y_pred, y_true)

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
  if i % 50 == 0:
    print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # through the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

print(f'*** Train data loss[{epochs}]: {loss}')

# Save our NN Model
torch.save(model.state_dict(), 'pca3_FLD_3.pt')

out = model.forward(X)
print(f'*** [out]: {out.shape}')
preds = torch.argmax(out, dim=1).numpy()

n_total = len(y)
idx_ok = np.where(preds == y)[0]
n_ok = len(idx_ok)
idx_err = np.where(preds != y)[0]
n_err = len(idx_err)

accuracy = n_ok / n_total
print(f"*** Trained Accuracy : {n_ok} / {n_total} -> {accuracy * 100:.2f}%")

# ==============================
# Re-Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_pred = model.forward(X)

  if FG_MSE_LOSS:
    loss = criterion(y_pred, y_test)
  else:
    loss = criterion(y_pred, y_true)

print(f'*** Final achieved loss: {loss} after {epochs} epochs')

print(f"W3 = \n {model.out.weight.data.numpy()}")
print(f"b3 = {model.out.bias.data.numpy()}")
