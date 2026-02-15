import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.preprocessing import StandardScaler

from etc_utils.viz_bloks import smart_show
# from levels.start_here.baseline_classifier import markersize

# 1. Data & LDA Projection (The better statistical input)
iris = load_iris()
X, y = iris.data, iris.target
X_std = StandardScaler().fit_transform(X)

# Note: LDA only provides 2 components for 3 classes.
# We'll use 2D LDA but visualize it effectively.
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_std, y)

# 2. Fisher Planes in LDA Space
def get_plane_2d(X_p, y_p, c1, c2):
    m1 = np.mean(X_p[np.isin(y_p, c1)], axis=0)
    m2 = np.mean(X_p[np.isin(y_p, c2)], axis=0)
    sw = np.dot((X_p[np.isin(y_p, c1)]-m1).T, (X_p[np.isin(y_p, c1)]-m1)) + \
         np.dot((X_p[np.isin(y_p, c2)]-m2).T, (X_p[np.isin(y_p, c2)]-m2))
    w = np.linalg.solve(sw, (m1 - m2))
    return w, -0.5 * np.dot(w, (m1 + m2))

w_set, b_set = get_plane_2d(X_lda, y, [0], [1, 2])
w_vv, b_vv   = get_plane_2d(X_lda, y, [1], [2])

# 3. Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_view = fig.add_subplot(111)  # , projection='3d'

# We use a dummy 3rd axis for 3D effect, or just plot 2D LDA
for i, color, name in zip([0, 1, 2], ['r', 'c', 'b'], iris.target_names):
    ax.scatter(X_lda[y == i, 0], X_lda[y == i, 1], ## np.zeros_like(X_lda[y == i, 0]),
               c=color, label=name, edgecolors='k')

# Drawing the V-V Plane (as a line in 2D projected into 3D)
x_vals = np.linspace(X_lda[:,0].min(), X_lda[:,0].max(), 10)
y_vals = -(w_vv[0] * x_vals + b_vv) / w_vv[1]

y_set = -(w_set[0] * x_vals + b_set) / w_set[1]
ax.plot(x_vals, y_vals, 'k--', lw=2, label='V-V Separator')
ax.plot(x_vals, y_set, 'r--', lw=2, label='Setosa Separator')

ax.set_title("Iris LDA Projection with Fisher Separator")

plt.ylim(1.5*X_lda[:,1].min(), 1.5*X_lda[:,1].max())
plt.legend()

# 4. Success Check
print(f"FLD V-V Coefficients: {w_vv}, {b_vv}")

W1 = lda.coef_
b1 = lda.intercept_

# Access fitted values
print("LDA Coefficients: \n", W1)
print("LDA Intercept: \n", b1)
print("Class Means: \n", lda.means_)

mapping = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}

def test_LDA(X_std, y, showErrors=False):
    y_pred = np.zeros(y.shape)
    total = 0
    correct = 0
    in_correct = []
    for i, data in enumerate(X_std):
        y_val = W1 @ data + b1
        y_pred[i] = y_val.argmax().item()

        # Correct or not
        total += 1
        if y_pred[i] == y[i]:
          correct += 1
        else:
          in_correct.append(i)
          if showErrors:
            # Will tell us what type of flower class our network thinks it is
            print(f'Point {i+1}. {mapping[y[i]]} --> {mapping[y_pred[i]]} (Incorrect!)')

    return correct, total, y_pred, in_correct

correct, total, y_pred, in_correct = test_LDA(X_std, y, showErrors=True)
print(f'We got {correct} RIGHT, out of {total}!')

for j in in_correct:
    ax.plot(X_lda[j, 0], X_lda[j, 1],
            markersize=20, marker='s', markeredgecolor='r', markerfacecolor='none')

# plt.show()
smart_show.smart_show(fgSaveFigures=True,
                      selectedFigures={1: "LDA_01"})

# exit # return

txt = f"""
# >>> import levels.da_Pretrained_Iris.Refined_3 as R3W

import numpy as np

# --- (fc1.weight) ([4, 3]) ---
W1 = np.array(\\
{np.array2string(W1, separator=', ')}
)

# --- (fc1.bias) ([3]) ---
b1 = np.array(\\
{np.array2string(b1, separator=', ')}
)
"""
with open("Refined_3.py", "w") as file:
    file.write(txt)
