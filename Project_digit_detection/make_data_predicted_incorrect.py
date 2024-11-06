import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt


# Load the saved model in .keras format
model = load_model('digit_classifier.keras')

# Load and preprocess the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., np.newaxis]  # Expand dimensions for grayscale images
def display_incorrect_predictions():
    # Get model predictions on the test set
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Find where the model's predictions don't match the true labels
    incorrect_indices = np.where(predicted_labels != y_test)[0]
    
    # Create a directory to save incorrect predictions
    incorrect_output_dir = 'incorrect_predictions'
    os.makedirs(incorrect_output_dir, exist_ok=True)

    # Counter for saved images
    count = 0
    max_images_to_save = 10

    # Loop over the incorrect predictions and save them
    for idx in incorrect_indices:
        if count >= max_images_to_save:
            break  # Stop after saving 10 images
        
        plt.imshow(x_test[idx].squeeze(), cmap='gray')  # Display image
        plt.title(f"True Label: {np.argmax(y_test[idx])}, Predicted: {predicted_labels[idx]}")
        plt.axis('off')  # Turn off axis numbers and ticks
        
        # Save the image
        plt.savefig(os.path.join(incorrect_output_dir, f'image_predicted_true_{np.argmax(y_test[idx])}_pred_{predicted_labels[idx]}.png'), bbox_inches='tight')
        plt.close()  # Close the plot to avoid display

        count += 1  # Increment the count

    print(f"Save incorrect predictions to '{incorrect_output_dir}' directory.")

display_incorrect_predictions()