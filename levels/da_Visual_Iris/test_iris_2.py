from etc_utils.data_harvest_io import load_uci_data
import levels.da_Statistical_Iris.hot_weights_4 as HW

import pandas as pd
import numpy as np

WEIGHTS = [HW.W1, HW.W2, HW.W3]
BIASES = [HW.b1, HW.b2, HW.b3]

U_setosa = [50, 34, 15, 2]
U_versicolor = [59, 28, 43, 13]
U_virginica = [66, 30, 56, 20]

from sklearn.preprocessing import StandardScaler

# X = df[features]
# y = df['variety']

X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)
species_names = y_labels.unique().tolist()

print( f"species_names: [{species_names}]" )
X_std = StandardScaler().fit_transform(X)

# Baseline ReLU Forward Pass
iTest = 1
# base_input = X_std[iTest]

species = species_names[iTest]

subset = X_std[y_labels == species]  ## class
base_input = subset.mean(axis=0) ## .values

acts = [base_input]
curr = base_input
for w, b in zip(WEIGHTS, BIASES):
    curr = np.maximum(0, np.dot(w, curr) + b)
    acts.append(curr)

for i in range(3):
    layer_vals = acts[i + 1]

yPred = acts[3] 

print( f"<{species}> sample: yPred = {yPred}" )

print( f"ALL layers: acts = " )
for i in range(4):
    print( f"{i}: [{acts[i]}]" )