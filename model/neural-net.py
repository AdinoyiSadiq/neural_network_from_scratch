import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

  # Layer initialization
  def __init__(self, n_inputs, n_neurons):
    # Initialize weights and biases
    # Note: We use 0.01 to multiply the gaussian distribution of weights to generate numbers that are a couple of magnitudes smaller.
    # Otherwise, the model will take more time to fit the data during the training process as starting values will be disproportionately 
    # large compared to the updates being made during training. 
    # The idea here is to start a model with non-zero values small enough that they won’t affect training.
    # This way, we have a bunch of values to begin working with, but hopefully none too large or as zeros. You can experiment with values other than 0.01
    # Note: weights would have n_inputs as rows and n_neurons as columns.
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    # Note: We’ll initialize the biases with the shape of (1, n_neurons), as a row vector, 
    # which will let us easily add it to the result of the dot product later, without additional operations like transposition.
    self.biases = np.zeros((1, n_neurons))
      
  # Forward pass
  def forward(self, inputs):
    # Calculate output values from inputs, weights and biases 
    self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])

# Note: In the output, you can see we have 5 rows of data that have 3 values each. 
# Each of those 3 values is the value from the 3 neurons in the dense1 layer after passing in each of the samples. 
'''
[[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
 [-1.0475188e-04  1.1395361e-04 -4.7983500e-05]
 [-2.7414842e-04  3.1729150e-04 -8.6921798e-05]
 [-4.2188365e-04  5.2666257e-04 -5.5912682e-05]
 [-5.7707680e-04  7.1401405e-04 -8.9430439e-05]]
'''
