import numpy as np

# Dense layer
class Layer_Dense:

  # Layer initialization
  def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
    # Initialize weights and biases
    # Note: We use 0.01 to multiply the gaussian distribution of weights to generate numbers that are a couple of magnitudes smaller.
    # Otherwise, the model will take more time to fit the data during the training process as starting values will be disproportionately 
    # large compared to the updates being made during training. 
    # The idea here is to start a model with non-zero values small enough that they won’t affect training.
    # This way, we have a bunch of values to begin working with, but hopefully none too large or as zeros. You can experiment with values other than 0.01
    # Note: weights would have n_inputs as rows and n_neurons as columns.
    self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
    # Note: We’ll initialize the biases with the shape of (1, n_neurons), as a row vector, 
    # which will let us easily add it to the result of the dot product later, without additional operations like transposition.
    self.biases = np.zeros((1, n_neurons))

    # Set regularization strength
    self.weight_regularizer_l1 = weight_regularizer_l1
    self.weight_regularizer_l2 = weight_regularizer_l2
    self.bias_regularizer_l1 = bias_regularizer_l1
    self.bias_regularizer_l2 = bias_regularizer_l2
  
  # Backward pass
  def backward(self, dvalues):
    
    # Gradients on parameters
    self.dweights = np.dot(self.inputs.T, dvalues) 
    self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

    # Gradients on regularization
    # L1 on weights
    if self.weight_regularizer_l1 > 0:
      dL1 = np.ones_like(self.weights)
      dL1[self.weights < 0] = -1
      self.dweights += self.weight_regularizer_l1 * dL1
    
    # L2 on weights
    if self.weight_regularizer_l2 > 0:
      self.dweights += 2 * self.weight_regularizer_l2 * self.weights
    
    # L1 on biases
    if self.bias_regularizer_l1 > 0:
      dL1 = np.ones_like(self.biases)
      dL1[self.biases < 0] = -1
      self.dbiases += self.bias_regularizer_l1 * dL1

    # L2 on biases
    if self.bias_regularizer_l2 > 0:
      self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
    
    # Gradient on values
    self.dinputs = np.dot(dvalues, self.weights.T)
      
  # Forward pass
  def forward(self, inputs, training):
    
    # Remember input values
    self.inputs = inputs
    # Calculate output values from inputs, weights and biases 
    self.output = np.dot(inputs, self.weights) + self.biases

  # Retrieve layer parameters
  def get_parameters(self):
    return self.weights, self.biases

  # Set weights and biases in a layer instance
  def set_parameters(self, weights, biases): 
    self.weights = weights
    self.biases = biases


# Dropout
class Layer_Dropout:

  # Init
  def __init__(self, rate):
    # Store rate, we invert it as for example for dropout 
    # of 0.1 we need success rate of 0.9
    self.rate = 1 - rate
      
  # Forward pass
  def forward(self, inputs, training):
    # Save input values
    self.inputs = inputs

    # If not in the training mode - return values
    if not training:
      self.output = inputs.copy()
      return

    # Generate and save scaled mask
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate 

    # Apply mask to output values
    self.output = inputs * self.binary_mask
  
  # Backward pass
  def backward(self, dvalues):
    # Gradient on values
    self.dinputs = dvalues * self.binary_mask


# Input "layer"
class Layer_Input:
  
  # Forward pass
  def forward(self, inputs, training): 
    self.output = inputs