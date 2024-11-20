import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# Load the saved model in .keras format
model = load_model('digit_classifier.keras')

# Load and preprocess the MNIST dataset
# (_, _), (x_test, y_test) = mnist.load_data()
# x_test = x_test / 255.0
# x_test = x_test[..., np.newaxis]  # Expand dimensions for grayscale images

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


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


# Load the saved model in .keras format
model = load_model('digit_classifier.keras')

# Preprocess function for a single custom image
def preprocess_custom_image(image_path):
    try:
        # Load the image
        original_img = Image.open(image_path)
        gray_img = original_img.convert('L')  # Convert to grayscale

        # Invert the image to ensure black background and white text
        inverted_img = ImageOps.invert(gray_img)
        binary_img = inverted_img.point(lambda p: 255 if p > 90 else 0)  # Thresholding

        # Resize the image to 28x28 (MNIST format)
        resized_img = binary_img.resize((28, 28))

        # Normalize the image for the model
        img_array = np.array(resized_img) / 255.0  # Normalize to range [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        return img_array, original_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


# Function to predict a single image
def predict_custom_image(image_path):
    img_array, original_img = preprocess_custom_image(image_path)

    if img_array is not None:
        # Show the original and processed images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_array.squeeze(), cmap='gray')
        plt.title("Processed Image for Model Input")
        plt.axis("off")
        plt.show()

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        print(f"Predicted Digit for {image_path}: {predicted_digit}")
    else:
        print(f"Skipping prediction for {image_path} due to processing error.")


# Function to process and predict multiple images in a directory
def predict_images_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # List all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {directory_path}.")
        return

    # Process and predict each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        print(f"\nProcessing image: {image_path}")
        predict_custom_image(image_path)


# Main function to predict multiple images in a directory
if __name__ == "__main__":
    # Directory containing the images
    # image_directory = "./data"
    
    # Predict all images in the directory
    predict_images_in_directory(image_directory)

# Test prediction
#predict_digit(1)

#display_incorrect_predictions();

