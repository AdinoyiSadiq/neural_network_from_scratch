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

# ReLU activation
class Activation_ReLU:
  
  # Forward pass
  def forward(self, inputs):
  
    # Calculate output values from inputs 
    self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

  # Forward pass
  def forward(self, inputs):

    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output # of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Make a forward pass through activation function # it takes the output of first dense layer here 
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs 
dense2.forward(activation1.output)

# Make a forward pass through activation function # it takes the output of second dense layer here 
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

# Note: In the output, you can see we have 5 rows of data that have 3 values each. 
# Each of those 3 values is the value from the 3 neurons in the dense1 layer after passing in each of the samples. 
'''
[[0.33333334 0.33333334 0.33333334]
 [0.3333332  0.3333332  0.33333364]
 [0.3333329  0.33333293 0.3333342 ]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
'''
