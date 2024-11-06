# Digit Classification 

This project use convolutional neural network to classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras, and the trained model can be used to predict the digit.

## Project Structute 

- `train_model.py` - Script to train the CNN model using the MNIST dataset and save it in the `.keras` format.
- `predict_model.py` - Script to load the trained model, make predictions on the test data, and display incorrect predictions.
- `digit_classifier.keras` - Saved model file (generated after training).

## Training Statistics

After training the model for 5 epochs on the MNIST dataset, here are the key statistics:

- **Epochs:** 5
- **Training Accuracy (Final):** 99.40% (0.9940)
- **Training Loss (Final):** 0.0182
- **Validation Accuracy (Final):** 98.82% (0.9882)
- **Validation Loss (Final):** 0.0406

### Validation Dataset

- **Number of Validation Samples:** 12,000 (20% of the training data was used for validation).
- **Validation Accuracy:** 98.82%
- **Validation Loss:** 0.0406

These metrics indicate that the model has achieved a high level of accuracy on both the training and validation datasets. The low training and validation loss suggests that the model has learned well without overfitting significantly.


## Dataset
The model is trained on the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

## Dependencies

To run this project, you need to install the following Python packages:

- `tensorflow`
- `numpy`
- `matplotlib`

Refer to the requirements.txt file to download the specified package versions.

## Model Summary 

The model processes grayscale images of handwritten digits (28x28 pixels) from the MNIST dataset. The architecture consists of two convolutional layers, followed by a fully connected dense layer, and finally a softmax output layer for classification.
Input Layer
Input Shape: (28, 28, 1)
Each image is 28x28 pixels in grayscale (1 channel).
Layer 1: Convolutional Layer 
Filter Count: 32 filters (or kernels), Filter Size: 5x5
Activation Function: ReLU (Rectified Linear Unit)
Output Shape: (24, 24, 32)
The image dimensions reduce from 28x28 to 24x24 because of the 5x5 filters (with no padding).
32 different feature maps (one for each filter) are generated from the input.
Layer 2: Max Pooling 
Pooling Size: 2x2, Stride: 2
Output Shape: (12, 12, 32)
This layer reduces the height and width of the feature maps by a factor of 2, so from 24x24 to 12x12, while keeping the 32 feature maps.
Layer 3: Convolutional Layer 
Filter Count: 32 filters, Filter Size: 5x5
Activation Function: ReLU
Output Shape: (8, 8, 32)
The convolution reduces the feature map size again to 8x8 by applying the 5x5 filters.
Layer 4: Max Pooling
Pooling Size: 2x2, Stride: 2
Output Shape: (4, 4, 32)
The max pooling layer halves the spatial dimensions again, resulting in 32 feature maps of size 4x4.
Layer 5: Flatten Layer
Output Shape: (512,)
The 32 feature maps, each of size 4x4, are flattened into a single 1D vector of size 512 (32 * 4 * 4 = 512).
Layer 6: Dense (Fully Connected) Layer
Units (Nodes): 128
Activation Function: ReLU
Output Shape: (128,)
This layer connects all 512 input features to 128 nodes with ReLU activation.
Layer 7: Output Layer
Units (Nodes): 10 (one for each digit from 0 to 9)
Activation Function: Softmax
Output Shape: (10,)
The softmax function outputs a probability distribution across the 10 possible digit classes.

