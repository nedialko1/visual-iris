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

    def forward_1(self, x):
        x = torch.relu(self.h1(x))
        x = x.reshape(1, 3)
        x = torch.relu(self.h2(x))
        return self.out(x)

# --- Test Run ---
model = IrisLogicNet(pca.components_, w_set, b_set, w_vv, b_vv)

# Convert y labels to tensors long
y_true = torch.tensor(y, dtype=torch.long)
X = torch.tensor(X, dtype=torch.float32)

preds = torch.argmax(model.forward(X), dim=1).numpy()

n_total = len(y)
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
ns = 3
y_value =  np.zeros((n_total,3))
## -100  ### -1 : much worse! Ratio 32/150

for iTest in  range(ns):  # [1]:  #
  iiThisClass = np.where(y == iTest)[0]
  print(f"*** iiThisClass.shape = {iiThisClass.shape}")
  ## y_value[iiThisClass,iTest] = 1 ## 10
  yThis = np.zeros((1,3))
  yThis[0,iTest] = 1
  N_REPLICATE = len(iiThisClass)
  y_value[iiThisClass] = np.tile(yThis, (N_REPLICATE, 1))

y_test = torch.tensor(y_value, dtype=torch.float32)

# idx = [ i + 50*j for i in range(5) for j in range(ns) ]
idx = [ i for i in range(5)  ]
for iTest in  range(ns):
    iiThisClass = np.where(y == iTest)[0]
    print(f"{iTest}. <> \n {y_value[iiThisClass[idx]]} ")

# ==============================
print(f"*** X.shape = {X.shape}")
print(f"*** y_test.shape = {y_test.shape}")

total = 0
correct = 0
with torch.no_grad():
  for i, data in enumerate(X):
    if i == 0:
      print(f"*** data.shape = {data.shape}")
    y_val =model.forward_1(data)

    # Correct or not
    total += 1
    if y_val.argmax().item() == y_test[i].argmax().item():
        # if y_val.argmax().item() == y_true[i]:
        correct += 1
    # else:
        # Will tell us what type of flower class our network thinks it is
        # print(f'{i+1}.) {y_true[i]} \t {y_val.numpy()} \t {y_val.argmax().item()}')

print(f'Zero-Training Testing: We got {correct} RIGHT, out of {total}!')

# -----------------------------
# Train our model!

target_layer_prefix = "out."

for name, param in model.named_parameters():
    if target_layer_prefix in name:

        param.requires_grad = True
        """
        if 'weight' in name:
            param.requires_grad = True
        else:  
            param.requires_grad = False
        """
    else:
        param.requires_grad = False

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.MSELoss() # CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

# Epochs? (one run through all the training data in our network)
epochs =   700 #  500   ## 300  ## wrong TARGET! 1300
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_test)  # Find the loss or error
  # loss = criterion(y_pred, y_true) # predicted values vs the y_train

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
torch.save(model.state_dict(), 'pca3_FLD_2.pt')

# ==============================
total = 0
correct = 0
with torch.no_grad():
  for i, data in enumerate(X):
    y_val =model.forward_1(data)
    # ? = np.expand_dims(x, axis=0)

    # Correct or not
    total += 1
    if y_val.argmax().item() == y_test[i].argmax().item():
    # if y_val.argmax().item() == y_true[i]:
      correct += 1
    # else:
      # Will tell us what type of flower class our network thinks it is
      # print(f'{i+1}.) {y_true[i]} \t {y_val.numpy()} \t {y_val.argmax().item()}')

print(f'Training: We got {correct} RIGHT, out of {total}!')

# ==============================
# Re-Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(X)  # X_test are features from our test set, y_eval will be predictions

  loss = criterion(y_eval, y_test)  # Find the loss or error
  # loss = criterion(y_eval, y_true)

print(f'*** Final achieved loss: {loss} after {epochs} epochs')

print(f"W3 = \n {model.out.weight.data.numpy()}")
print(f"b3 = {model.out.bias.data.numpy()}")

# ==================================
# Graph it out!

import matplotlib.pyplot as plt

plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel('Epoch')

plt.show()

