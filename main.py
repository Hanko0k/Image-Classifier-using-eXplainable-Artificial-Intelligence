import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import TensorBoard
import datetime


# Directory paths
base_dir = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset"
train_dir = f'{base_dir}/train'
validation_dir = f'{base_dir}/validation'

def normalize(image, label):
    """
    Normalize the images to [0, 1] range.
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label




def load_data(image_size=(64, 64), batch_size=32):
    """
    Load and preprocess image data from directory.
    
    Parameters:
    - dataset_path: str, path to the dataset directory.
    - image_size: tuple, the size to which each image is resized.
    - batch_size: int, the number of images to process in each batch.
    
    Returns:
    - train_dataset: training dataset.
    - validation_dataset: validation dataset.
    """
    # Load training data
    train_dataset = image_dataset_from_directory(
        directory=f"{base_dir}/train",
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        label_mode='binary'  # Because we have binary classification
    )
    
    # Load validation data
    validation_dataset = image_dataset_from_directory(
        directory=f"{base_dir}/validation",
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        label_mode='binary'  # Because we have binary classification
    )

        # Load validation data
    test_dataset = image_dataset_from_directory(
        directory=f"{base_dir}/test",
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        label_mode='binary'  # Because we have binary classification
    )
    
    return train_dataset, validation_dataset, test_dataset

train_images, val_images, test_images = load_data()

# Assuming train_dataset and validation_dataset are already defined
train_dataset = train_images.map(normalize)
validation_dataset = val_images.map(normalize)
test_dataset = test_images.map(normalize)  # Normalize test data similarly if you have a test dataset



# Define the CNN model
model = Sequential([
    # Convolutional layer with 32 filters, a kernel size of 3, activation function ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Adjust input_shape based on your data
    MaxPooling2D(2, 2),
    # Another convolutional layer, increasing depth
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # Flatten the output of the conv layers to feed into the dense layer
    Flatten(),
    # Dense layer for prediction
    Dense(128, activation='relu'),
    # Dropout for regularization
    Dropout(0.5),
    # Output layer with sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up TensorBoard logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(train_images, epochs=10, validation_data=val_images)

# Save the model
model.save('basic_cnn_model.h5')

test_loss, test_accuracy = model.evaluate(test_images)

print(test_loss)
print(test_accuracy)