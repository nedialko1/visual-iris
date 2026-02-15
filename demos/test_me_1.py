"""
** test_me.py **
	a  script to QA/health-test most of the baseplate utilities
"""

# import torch  # imported implicitly via test_run & train_me below

import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models_io import baseline_MLP, save_model_params
from etc_utils.data_harvest_io import load_uci_data

from etc_utils.viz_bloks import smart_show

from levels.start_here.test_run import *
from levels.start_here.train_me import *

# ==============================
def test_me():

	#1. Load the **baseline** iris MLP model & the related iris-flowers benchmark data:
	model = baseline_MLP.model
	X, y, y_labels = load_uci_data.load_uci_data(returnData=True, plotData=False)

	# Train Test Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

	y_train = torch.LongTensor(y_train.numpy())
	y_test = torch.LongTensor(y_test.numpy())

	#2. Visualize the iris-flowers benchmark data:
	fignum = load_uci_data.plot_input_data(X_test, y_test)

	# Right away test run the baseline iris-flowers classifier with its random weights:
	layers_dict = model.state_dict()
	save_model_params.save_model_params(
		model_name="Baseline_MLP_init", layers_dict=layers_dict,
		version=0, out_dir="models_io/saved_model_pars")

	num_epochs = 100
	criterion = baseline_MLP.criterion
	optimizer = baseline_MLP.optimizer
	losses = train_me(X_train, y_train, model, criterion, optimizer, epochs=num_epochs)

	# Graph it out!
	plt.figure(11)
	plt.plot(range(num_epochs), losses)
	plt.ylabel("loss/error")
	plt.xlabel('Epoch')

	# ==================================
	# Evaluate Model on Test Data Set (validate model on test set)
	with torch.no_grad():  # Basically turn off back propogation
		y_eval = model.forward(X_test)  # X_test are features from our test set, y_eval will be predictions
		loss = criterion(y_eval, y_test)  # Find the loss or error

	print(f"After training: loss[{num_epochs}] = {loss} ")

	correct, total, y_pred = test_run(model, X_test, y_test, showErrors=False)
	print(f'We got {correct} RIGHT, out of {total}!')

	iiError = np.where(y_pred.numpy() != y_test.numpy())[0]
	load_uci_data.plot_input_data(X_test[iiError], y_pred[iiError], fgErrors=True, fignum=fignum)

	layers_dict = model.state_dict()
	save_model_params.save_model_params(model_name="Baseline_MLP_trained",
										layers_dict=layers_dict, version=1,
										out_dir="models_io/saved_model_pars")

	smart_show.smart_show(fgSaveFigures=True,
			   selectedFigures={11: "MLP_train_performance_evolution",
								1: "baseline_MLP_trained_performance"})

if __name__ == "__main__":
	test_me()