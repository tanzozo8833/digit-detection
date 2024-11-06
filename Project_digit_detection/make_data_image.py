import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Create a directory to save the images
output_dir = 'mnist_images'
os.makedirs(output_dir, exist_ok=True)

# Save the first 10 images
for i in range(10):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Label: {y_train[i]}')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.savefig(os.path.join(output_dir, f'image_{i+1}_label_{y_train[i]}.png'), bbox_inches='tight')
    plt.close()  # Close the plot to avoid display

print("10 images have been saved to the 'mnist_images' directory.")

