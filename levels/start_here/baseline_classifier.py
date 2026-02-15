import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from etc_utils.data_harvest_io import load_uci_data

from etc_utils.viz_bloks import smart_show
fig_version = 0

if fig_version == 0:
    from rand_weights.baseline_weights_rand import weights
elif fig_version == 1:
    from rand_weights.baseline_weights_1 import weights
else:
    from rand_weights.baseline_weights_middle import weights


def relu(x):
    # x = np.asarray(x)
    y = np.maximum(0, x)
    return y

def softmax(x):
    # x = np.asarray(x)
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    y = exps / np.sum(exps, axis=1, keepdims=True)
    return y

# 1. Load Data
X, y_true = load_uci_data.load_uci_data(returnData=True, plotData=False)

# 2. Forward Pass (4-8-9-3)
W1, b1, W2, b2, W3, b3 = weights

# X is a Torch tensor!!!
# The following op makes the whole lin-alg daisy-chain happy!
X =  np.asarray(X)

h1 = relu(X @ W1 + b1)
h2 = relu(h1 @ W2 + b2)
logits = h2 @ W3 + b3

y_pred = np.argmax(softmax(logits), axis=1)

# 3. Visualization logic
styles = {
    0: {'color': 'blue', 'marker': 'o', 'label': 'Setosa'},
    1: {'color': 'cyan', 'marker': '^', 'label': 'Versicolor'},
    2: {'color': 'green', 'marker': 's', 'label': 'Virginica'}
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Diagnostic Logic Output
print("\n--- Classification Diagnostic ---")
print("NOTE: All points are plotted using the symbol of their PREDICTED class.")
print("Colors/Shapes Logic:")
print("- Correct Classifications: Standard color (Blue/Cyan/Green).")
print("- Misclassifications: RED+hollow (indicates the system was 'confident but wrong').")
print("\nExample Interpretation:")
print("- A RED SQUARE: A data point the model misidentified as Virginica (Square).")
print("- A RED '^': A data point the model misidentified as Versicolor ('^').")

# y_true is a Torch tensor!!!
y_true =  np.asarray(y_true)

n_ERRORS = np.where( y_true != y_pred )[0].shape[0]

for i in range(len(X)):
    true_cls = y_true[i]
    pred_cls = y_pred[i]

    # Logic: Symbol reflects PREDICTION; Color reflects ACCURACY
    marker = styles[pred_cls]['marker']
    color = styles[true_cls]['color']  if true_cls == pred_cls else 'none'
    markersize = 60 if true_cls == pred_cls else 100
    tag = str(i)  if true_cls != pred_cls else None

    if n_ERRORS < 10:
        ax1.text(X[i, 0], X[i, 1], tag, fontsize=20)
    ax1.scatter(X[i, 0], X[i, 1], c=color,
                s=markersize, marker=marker, edgecolors=color if true_cls == pred_cls else 'red')

    if n_ERRORS < 10:
        ax2.text(X[i, 2], X[i, 3], tag, fontsize=20)
    ax2.scatter(X[i, 2], X[i, 3], c=color,
                s=markersize, marker=marker, edgecolors=color if true_cls == pred_cls else 'red')

# Keep the axes labels as requested
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')

plt.tight_layout()

# --------------------------------------

print("="*80)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm,
                     index=['Actual Setosa', 'Actual Versicolor', 'Actual Virginica'],
                     columns=['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

print("\n--- System Performance: 'The Uniformly Imperfect' Baseline ---")
print(cm_df)
print("\nTake-away: Note how the 'RED' markers are distributed across all three symbols.")

figureName = "baseline_MLP_classification"
figureName = figureName+"_trained" if fig_version == 1 else figureName+"_UN_trained"

if fig_version > 1:
    plt.show()
else:
    smart_show.smart_show(fgSaveFigures=True, selectedFigures={1: figureName})