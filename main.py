import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()      # https://keras.io/api/datasets/fashion_mnist/ fashion_mnist documentation

# Class names - the index of the class names corresponds to the class label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to plot one sample image for each class
'''
def plot_samples(x, y, class_names):
    num_classes = len(class_names)
    fig, axes = plt.subplots(1, num_classes, figsize=(20, 5))
    for i in range(num_classes):
        idx = np.where(y == i)[0][0]
        axes[i].imshow(x[idx], cmap='gray')
        axes[i].set_title(class_names[i])
        axes[i].axis('off')
    plt.show()
'''
# Plot one sample image for each class
#plot_samples(x_train, y_train, class_names)

print(x_train[0][15])