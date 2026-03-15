# Multi-Layer Neural Network from Scratch

A multilayer neural network implemented in plain Python and NumPy — no ML libraries. Built as a final project for a graduate machine learning course at Fordham University.

The goal was to understand what's happening inside a neural network by building it from the ground up: forward pass, backpropagation, weight updates, and a custom pruning mechanism that suppresses weak connections during training.

---

## Features

- Configurable number of hidden layers and neurons per layer
- Sigmoid activation function
- Backpropagation with adjustable learning rate
- **Neural suppression (pruning):** weights that fall below a threshold are zeroed out at a specified interval during training, simulating the way biological neurons prune weak connections
- Outputs a graph of error rate over epochs
- Outputs confusion matrices for both training and test data

---

## Current limitations

- The final output layer is fixed at one neuron, so the network currently handles binary classification only. Extending this to multi-class output is straightforward but has not been implemented yet.

---

## Main function

```python
neural_net(
    _epochs,             # number of training epochs (int)
    _epsil,              # learning rate (float)
    _features,           # input features (numpy array)
    _classes,            # target classes (numpy array)
    _layers,             # hidden layer sizes, e.g. np.array([8, 4]) (numpy array)
    _supress_threshold,  # weights below this value will be pruned (float)
    _k_iter,             # how often (in epochs) to check for pruning (int)
)
```

**Returns:**
- Trained weights and bias weights
- Array of neuron values (used for evaluating test data)
- Array of error values per epoch

---

## Helper functions

- **`sig(x)`** — sigmoid activation function
- **`pruning(original, prune, supress_threshold, iter, k_iter)`** — zeroes out weights below the suppression threshold at every `k_iter` epochs. Called internally by `neural_net`; users don't need to call this directly.

---

## How to use

**1. Set your data**

Go to the section marked `Input any changes that need to be done`. Put your features in the `features` array and your classes in the `classes` array. This example has classes in column 0 — adjust the column index if your data is different. Do the same in the `checking test data` section for your test set.

**2. Set your hyperparameters**

Go to the section marked `USER INPUT` and set:

| Parameter | Description |
|---|---|
| `_epochs` | How many times to run through the training data |
| `_epsil` | Learning rate — smaller values learn more slowly but more stably |
| `_layers` | Hidden layer sizes as a numpy array, e.g. `np.array([8, 4])` |
| `_supress_threshold` | Weights below this value get pruned |
| `_k_iter` | Check for pruning every N epochs |

**3. Run**

Execute the script. You'll get:
- A plot of error rate over training epochs
- A confusion matrix on training data
- A confusion matrix on test data

---

## Requirements

```
numpy
matplotlib
```
