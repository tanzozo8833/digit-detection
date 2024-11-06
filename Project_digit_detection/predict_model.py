import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# Load the saved model in .keras format
model = load_model('digit_classifier.keras')

# Load and preprocess the MNIST dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., np.newaxis]  # Expand dimensions for grayscale images

# Preprocess the image(image in MNIST dataset) to predict (reshape and normalize) 
# def preprocess_image(image_index):
#     img = x_test[image_index]
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Predict the digit for a given image index
# def predict_digit(image_index):
#     img = preprocess_image(image_index)
#     prediction = model.predict(img)
#     predicted_digit = np.argmax(prediction)
    
#     # Display the image using matplotlib
#     plt.imshow(x_test[image_index].squeeze(), cmap='gray')  # Remove extra dimensions for display
#     plt.title(f'Image Index: {image_index}')
#     plt.show()

#     # Print the predicted digit
#     print(f'Predicted Digit: {predicted_digit}')

# def display_incorrect_predictions():
#     # Get model predictions on the test set
#     predictions = model.predict(x_test)
#     predicted_labels = np.argmax(predictions, axis=1)
    
#     # Find where the model's predictions don't match the true labels
#     incorrect_indices = np.where(predicted_labels != y_test)[0]
    
#     # Loop over the incorrect predictions and display them
#     for idx in incorrect_indices:
#         plt.imshow(x_test[idx].squeeze(), cmap='gray')  # Display image
#         plt.title(f"True Label: {y_test[idx]}, Predicted: {predicted_labels[idx]}")
#         plt.show()
image_path = "my_image_input/my_image_3.jpeg"

def predict_custom_image(image_path):
    # Load the image
    img1 = Image.open(image_path)
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    # Invert image with absolutely black background, white text
    img = ImageOps.invert(img)
    img = img.point(lambda p: 255 if p > 128 else 0)  # Threshold at 128
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    
    # Preprocess the image
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Make a prediction
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Original Image")
    
    # Show the processed image
    plt.subplot(1, 2, 2)
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title("Processed Image for Model Input")
    plt.show()
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    print(f'Predicted Digit: {predicted_digit}')
# Call the function to predict your custom image
predict_custom_image(image_path)


# Test prediction
#predict_digit(1)

#display_incorrect_predictions();

