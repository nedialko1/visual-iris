"""
** test_run.py **
This function is test runs any given iris-flowers classifiers model
"""

import torch

def test_run(model, X_test, y_test, showErrors=False):

    total = 0
    correct = 0
    y_pred = torch.zeros(y_test.shape)
    with torch.no_grad():
      for i, data in enumerate(X_test):
        y_val = model.forward(data)

        if y_test[i] == 0:
          x = "Setosa"
        elif y_test[i] == 1:
          x = 'Versicolor'
        else:
          x = 'Virginica'

        # Correct or not
        total += 1
        y_pred[i] = y_val.argmax().item()
        if y_pred[i] == y_test[i]:
          correct += 1
        elif showErrors:
          # Will tell us what type of flower class our network thinks it is
          print(f'Point {i+1}. *** class {x}: *** {y_test[i]} --> {y_pred[i]} (Incorrect!)')

    return correct, total, y_pred       
    