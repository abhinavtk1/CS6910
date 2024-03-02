import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()      # fashion_mnist documentation - https://keras.io/api/datasets/fashion_mnist/ 

# Class names - the index of the class names corresponds to the class label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Prepare the data
print(x_train[0][15])