import numpy as np
import nnfs
from nnfs.datasets import sine_data, spiral_data, vertical_data
from accuracy.accuracy import Accuracy_Categorical
from activation.activation import Activation_ReLU, Activation_Softmax
from data_manager.data_manager import create_data_mnist
from layer.layer import Layer_Dense
from loss.loss import Loss_CategoricalCrossentropy
from model import Model
from optimizer.optimizer import Optimizer_Adam

nnfs.init()

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5


# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
  loss=Loss_CategoricalCrossentropy(),
  optimizer=Optimizer_Adam(decay=1e-3),
  accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

# Retrieve and print parameters
parameters = model.get_parameters()

# New model
# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss and accuracy objects
# We do not set optimizer object this time - there's no need to do it
# as we won't train the model
model.set(
loss=Loss_CategoricalCrossentropy(),
accuracy=Accuracy_Categorical() )

# Finalize the model
model.finalize()

# Set model with parameters instead of training it
model.set_parameters(parameters)

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the model
model.save('fashion_mnist.model')

# Load the model
model = Model.load('fashion_mnist.model')
  
# Evaluate the model
print('Saved model evaluation')
model.evaluate(X_test, y_test)

# Predict on the first 5 samples from validation dataset
# and print the result
print('Predict on test data')
confidences = model.predict(X_test[:5])
predictions = model.output_layer_activation.predictions(confidences)
print(predictions)

fashion_mnist_labels = {
  0: 'T-shirt/top',
  1: 'Trouser',
  2: 'Pullover',
  3: 'Dress',
  4: 'Coat',
  5: 'Sandal',
  6: 'Shirt',
  7: 'Sneaker',
  8: 'Bag',
  9: 'Ankle boot'
}

for prediction in predictions:
  print(fashion_mnist_labels[prediction])
