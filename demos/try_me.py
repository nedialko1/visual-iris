"""
** try_me.py **
This is the very first script to attempt 
once you successfully cloned the **visual-iris** repository
and fulfilled the python requirements by installing them 
Just do:
```
python  try_me.py
```
This will: 
* produce scatter plots of the input data points
* load the baseline iris-flowers classifier NN model (MLP) 
* use pretrained weights and classify the data points
* the results of the classification will be visualized 
	on a second set of scatter plots 
* mis-classified data points will be shown in red  
"""

import numpy as np
from sklearn.model_selection import train_test_split

from models_io import baseline_MLP
from etc_utils.data_harvest_io import load_uci_data

from etc_utils.viz_bloks import smart_show

from levels.start_here.test_run import *

# Seed 42 for reproducibility
from levels.etc_MNIST.seed_models import set_seed
set_seed(42)

# ==============================
def try_me():

	#1. Load the **baseline** iris MLP model & the related iris-flowers benchmark data:
	model = baseline_MLP.model
	X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)

	# Visualize the iris-flowers benchmark data:
	load_uci_data.plot_input_data(X, y)
	smart_show.smart_show(fgSaveFigures=True, selectedFigures={1: "load_uci_data"})

	# Train Test Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

	y_train = torch.LongTensor(y_train.numpy())
	y_test = torch.LongTensor(y_test.numpy())

	#2. Visualize the iris-flowers Test data:
	fignum = load_uci_data.plot_input_data(X_test, y_test)

	#3. Right away test run the baseline iris-flowers classifier with its random weights:
	correct, total, y_pred = test_run(model, X_test, y_test, showErrors=False)
	print(f'We got {correct} RIGHT, out of {total}!')

	import matplotlib.pyplot as plt

	iiError = np.where(y_pred.numpy() != y_test.numpy())[0]
	load_uci_data.plot_input_data(X_test[iiError], y_pred[iiError], fgErrors=True, fignum=fignum )

	plt.suptitle( f"Untrained Baseline MLP: Accuracy {100*correct/total:.2f}% ({correct} of {total})",
				  fontsize=16, y=0.98)
	# print(f"{plt.rcParams.keys()}")

	smart_show.smart_show(fgSaveFigures=True,
						  selectedFigures={1: "baseline_MLP_UN_trained_performance"})


if __name__ == "__main__":
    try_me()