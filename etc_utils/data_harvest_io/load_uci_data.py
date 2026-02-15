import pandas as pd
import os

import torch

import matplotlib.pyplot as plt
import seaborn as sns

from etc_utils.viz_bloks import smart_show

# --------------------------------

from pathlib import Path

# 1. Get the path of the current script file
script_path = Path(__file__).resolve()
# 2. Get the directory containing the script
script_dir = script_path.parent
# 3. Define the relative path to the subfolder and file
relative_subfolder_path = Path("iris_uci.csv")
# 4. Combine them to get the absolute path to the data file
absolute_file_path = script_dir / relative_subfolder_path

# --------------------------------------------------------

# Configuration
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
FILE_NAME = absolute_file_path
COLUMNS = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety']
ROWS = [ 'Setosa', 'Versicolor', 'Virginica' ]

markers = {'Setosa':'o', 'Versicolor':'v', 'Virginica':'s'}

def plot_input_data(X, yClass, fgErrors=False, fignum=1):
    Z = X.numpy()
    plot_data = {name: col for name, col in zip(COLUMNS, Z.T)}

    cc = yClass.numpy().astype(int)
    plot_data["variety"] = [ ROWS[x] for x in cc ]

    if fgErrors:
        plt.figure(num=fignum)
        axes = plt.gcf().get_axes()
        plot_class_errors(plot_data, axes)
    else:
        fig, axes = plt.subplots(1, 2, num=fignum, figsize=(14, 6))
        plot_all_input_data(plot_data, axes)
        plt.tight_layout()

    return fignum        

def plot_class_errors(plot_data, axes):
    ax1, ax2 = axes   
    
    ## markers = {'Setosa':'^', 'Versicolor':'p', 'Virginica':'h'}
    markerSize = 250

    sns.scatterplot(data=plot_data, x='sepal.length', y='sepal.width', ec='r', fc="none", \
        style="variety", s=markerSize, markers=markers, ax=ax1 )
    ax1.set_title("Sepal Dimensions")

    sns.scatterplot(data=plot_data, x='petal.length', y='petal.width', ec='r', fc="none", \
        style="variety", s=markerSize, markers=markers, ax=ax2 )
    ax2.set_title("Petal Dimensions")


def plot_all_input_data(plot_data, axes):
    ax1, ax2 = axes    

    custom_palette = {'Setosa':'b', 'Versicolor':'c', 'Virginica':'g'}

    markerSize = 100

    sns.scatterplot(data=plot_data, x='sepal.length', y='sepal.width', hue='variety',  \
        palette=custom_palette, style="variety", s=markerSize, markers=markers, ax=ax1 )
    ax1.set_title("Sepal Dimensions")

    sns.scatterplot(data=plot_data, x='petal.length', y='petal.width', hue='variety', \
        palette=custom_palette, style="variety", s=markerSize, markers=markers, ax=ax2 )
    ax2.set_title("Petal Dimensions")
    

def get_data(fileName=FILE_NAME):
    print(f"*** the Current Working Directory is: \n {Path.cwd()}")
    print(f"*** Looking up File {FILE_NAME} ...")
    if not os.path.exists(fileName):
        print(f"*** File {FILE_NAME} was not found!")

        print(f"Downloading data from {DATA_URL}...")
        df = pd.read_csv(DATA_URL, names=COLUMNS)
        df.to_csv(fileName, index=False)
    return pd.read_csv(fileName)

def load_uci_data(returnData=False, plotData=True):
    """
    If no source is provided, loads default data and plots it.
    If a source is provided, returns the loaded data.
    """

    df = get_data()  
    print(df.tail())

    X = torch.FloatTensor(df.drop('variety', axis=1).to_numpy() )

    # Change last column from strings to numbers
    y_true_labels = df['variety']
    y_unique_labels = y_true_labels.values
    mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    inv_mapping = {v: k for k, v in mapping.items()}
    y_true = torch.FloatTensor([mapping[label] for label in y_unique_labels])

    if plotData:
    # -----------------------------
    # Also Default behavior: show the data scatter plots:

        plot_input_data(X,y_true)
        plt.show()
        # smart_show.smart_show(fgSaveFigures=True, selectedFigures={1: "load_uci_data"})

    if returnData:    
    # -----------------------------
    # Specific behavior: Return the data 

        return X, y_true, y_true_labels

if __name__ == "__main__":
    load_uci_data()