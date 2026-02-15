# Train a model:
# Epochs? (one NN model inference for any training data)

import torch

def train_me(X_train, y_train, model, criterion, optimizer, epochs=100):

  losses = []
  for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train) # Get predicted results

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train) # predicted values vs the y_train

    # Keep Track of our losses
    losses.append(loss.detach().numpy())

    # print every 10 epoch
    if i % 10 == 0:
      print(f'Epoch: {i} and loss: {loss}')

    # Do some back propagation: take the error rate of forward propagation and feed it back
    # thru the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return losses

