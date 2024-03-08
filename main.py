import numpy as np
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()      # fashion_mnist documentation - https://keras.io/api/datasets/fashion_mnist/ 

# Class names - the index of the class names corresponds to the class label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Prepare the data
x_train = x_train.reshape(x_train.shape[0], -1) # change shape to (60000, 784)
x_train = x_train.astype('float32') / 255      # normalize the pixel values to [0, 1]

def init_weights():
    W1 = np.random.rand(10, 784)
    b1 = np.random.rand(10,1)
    W2 = np.random.rand(10, 10)
    b2 = np.random.rand(10,1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z),axis=0)
    return A

def fwd_prop(W1, b1, W2, b2):
    Z1 = W1.dot(x_train.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

W1, b1, W2, b2 = init_weights()
Z1, A1, Z2, A2 = fwd_prop(W1, b1, W2, b2)

print(sum(A2[0]))
