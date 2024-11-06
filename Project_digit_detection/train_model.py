import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Expand dimensions to include the channel (for grayscale images)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# To see how many samples you have:
# print(f"Number of training samples: {x_train.shape[0]}")
# print(f"Number of test samples: {x_test.shape[0]}")

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model using the Input layer as the first layer
def build_model():
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(28, 28, 1)))

    # Convolutional layer and max pooling
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Compile the model
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Save the model using the new .keras format
model.save('digit_classifier.keras')