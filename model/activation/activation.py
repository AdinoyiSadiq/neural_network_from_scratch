import numpy as np

# Linear activation
class Activation_Linear:
  
  # Forward pass
  def forward(self, inputs, training): 
    # Just remember values 
    self.inputs = inputs 
    self.output = inputs

  # Backward pass
  def backward(self, dvalues):
    # derivative is 1, 1 * dvalues = dvalues - the chain rule 
    self.dinputs = dvalues.copy()

  # Calculate predictions for outputs
  def predictions(self, outputs): 
    return outputs


# ReLU activation
class Activation_ReLU:
  
  # Forward pass
  def forward(self, inputs, training):
    
    # Remember input values
    self.inputs = inputs
    # Calculate output values from inputs 
    self.output = np.maximum(0, inputs)

  # Backward pass
  def backward(self, dvalues):
  
    # Since we need to modify original variable, 
    # let’s make a copy of values first 
    self.dinputs = dvalues.copy()
    # Zero gradient where input values were negative
    self.dinputs[self.inputs <= 0] = 0

  # Calculate predictions for outputs
  def predictions(self, outputs): 
    return outputs
  

# Softmax activation
class Activation_Softmax:

  # Forward pass
  def forward(self, inputs, training):
    # Remember input values 
    self.inputs = inputs

    # Get unnormalized probabilities
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

    # Normalize them for each sample
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    self.output = probabilities

      # Backward pass
  def backward(self, dvalues):

    # Create uninitialized array
    self.dinputs = np.empty_like(dvalues)

    # Enumerate outputs and gradients
    for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
      # Flatten output array
      single_output = single_output.reshape(-1, 1)
      # Calculate Jacobian matrix of the output
      jacobian_matrix = np.diagflat(single_output) - \
                        np.dot(single_output, single_output.T)

      # Calculate sample-wise gradient
      # and add it to the array of sample gradients
      self.dinputs[index] = np.dot(jacobian_matrix,
                                    single_dvalues)

  # Calculate predictions for outputs
  def predictions(self, outputs): 
    return np.argmax(outputs, axis=1)
  

# Sigmoid activation
class Activation_Sigmoid:
  
  # Forward pass
  def forward(self, inputs, training):
    # Save input and calculate/save output 
    # of the sigmoid function
    self.inputs = inputs
    self.output = 1 / (1 + np.exp(-inputs))

  # Backward pass
  def backward(self, dvalues):
    # Derivative - calculates from output of the sigmoid function 
    self.dinputs = dvalues * (1 - self.output) * self.output

  # Calculate predictions for outputs
  def predictions(self, outputs): 
    return (outputs > 0.5) * 1  


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

  # Backward pass
  def backward(self, dvalues, y_true):

    # Number of samples
    samples = len(dvalues)

    # If labels are one-hot encoded,
    # turn them into discrete values
    if len(y_true.shape) == 2:
        y_true = np.argmax(y_true, axis=1)

    # Copy so we can safely modify
    self.dinputs = dvalues.copy()
    # Calculate gradient
    self.dinputs[range(samples), y_true] -= 1
    # Normalize gradient
    self.dinputs = self.dinputs / samples