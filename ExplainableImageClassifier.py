import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Rescaling, RandomFlip, RandomRotation, GlobalAveragePooling2D, Dense, Input, Lambda, Layer, Reshape, BatchNormalization, Dropout
# from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from lime import lime_image
import shap
from skimage.segmentation import mark_boundaries
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# from imblearn.under_sampling import RandomUnderSampler
import os
from datetime import datetime
import json

# Define the center-biased attention layer
class CenterBiasedAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(CenterBiasedAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)
        
        # Create a center bias mask
        height, width = input_shape[1], input_shape[2]
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        center_bias = np.exp(-0.5 * (distance / (min(height, width) / 4))**2)
        self.center_bias = tf.convert_to_tensor(center_bias, dtype=tf.float32)
        self.center_bias = tf.expand_dims(self.center_bias, axis=-1)  # Shape: (height, width, 1)
        self.center_bias = tf.expand_dims(self.center_bias, axis=0)   # Shape: (1, height, width, 1)

    def call(self, inputs):
        # Tile the center bias to match the input tensor shape
        center_bias_expanded = tf.tile(self.center_bias, [tf.shape(inputs)[0], 1, 1, inputs.shape[-1]])
        
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        
        # Apply center bias to attention weights
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        attention_weights = tf.tile(attention_weights, [1, 1, 1, inputs.shape[-1]])
        attention_weights = attention_weights * center_bias_expanded
        
        # Normalize attention weights
        attention_weights_sum = tf.reduce_sum(attention_weights, axis=[1, 2], keepdims=True)
        attention_weights = attention_weights / attention_weights_sum
        
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=[1, 2])
        return context_vector, attention_weights

    def get_config(self):
        config = super(CenterBiasedAttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Register the custom layer for serialization
tf.keras.utils.get_custom_objects().update({
    "CenterBiasedAttentionLayer": CenterBiasedAttentionLayer
})

class SpatialAttentionLayer(Layer):
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()

    def build(self, input_shape):
        self.conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')

    def call(self, inputs):
        attention = self.conv1(inputs)
        attention = self.conv2(attention)
        return inputs * attention

# Define the attention layer
class AttentionLayer(Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class PreprocessInputLayer(Layer):
    def __init__(self, **kwargs):
        super(PreprocessInputLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Apply the preprocess_input from ResNet50 directly here
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(inputs)

    # def get_config(self):
    #     base_config = super(PreprocessInputLayer, self).get_config()
    #     return base_config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class ExplainableImageClassifier:
    def __init__(self, model=None, explainable_method=None):
 
        self.models = {}
        self.explainer = lime_image.LimeImageExplainer()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.normalization_layer = Rescaling(1./255)

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            monitor='val_loss',
            # min_delta=0.001,
            patience=3, # Small dataset of apples
            verbose=1,
            mode='min',
            restore_best_weights=True
        )


        return None
    
    # Function to visualize attention map
    def visualize_attention_map(self, model_name, image):
        attention_layer_model = Model(inputs=self.models[model_name].input,
                                            outputs=self.models[model_name].get_layer('center_biased_attention').output[1])
        attention_weights = attention_layer_model.predict(np.expand_dims(image, axis=0))
        attention_weights = np.squeeze(attention_weights, axis=0)  # Remove batch dimension
        attention_weights = np.mean(attention_weights, axis=-1)  # Average over channels

        # Plot the original image and the attention map
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)

        plt.subplot(1, 2, 2)
        plt.title('Attention Map')
        plt.imshow(attention_weights, cmap='viridis')
        plt.show()
    
    def load_pretrained_model(self, model, mods) -> None:
        if model == 'ResNet50':
            # Start by defining the input layer with the correct input shape
            input_tensor = Input(shape=(224, 224, 3))
            # Include the Rescaling layer right after the input
            # rescaled_input = Rescaling(scale=1./255)(input_tensor)

            # Apply the correct preprocessing using a Lambda layer
            # processed_input = Lambda(preprocess_input)(input_tensor)
            processed_input = PreprocessInputLayer()(input_tensor)

            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=processed_input)

            for layer in base_model.layers:
                layer.trainable = False

            if mods == 'binary':
                # Add custom layers on top of ResNet
                x = base_model.output
                x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
                x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
                predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

                # This is the model we will train
                self.model = Model(inputs=base_model.input, outputs=predictions)

                self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


        return None
    
    def train_loaded_model(self) -> None:
        train_ds, val_ds, test_ds = self._load_dataset(self.data_path)

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=40,
            callbacks=[self.early_stopping]
        )

        save_name = 'ResNet50_2'
        self.model.save(save_name + '.keras')

        # history = self.model.fit(
        #     train_generator,
        #     steps_per_epoch=train_generator.samples // train_generator.batch_size,
        #     epochs=10,  # You can change the number of epochs
        #     validation_data=validation_generator,
        #     validation_steps=validation_generator.samples // validation_generator.batch_size)

        return None
    
    def save_architecture(self, model_name):
        DIR_PATH = 'Model Architectures'
        if not os.path.exists(DIR_PATH):
            os.makedirs(DIR_PATH)
            
        model_json = self.models[model_name].to_json()
        json_path = os.path.join(DIR_PATH, model_name + '.json')
        with open(json_path, 'w') as file:
            json.dump(json.loads(model_json), file, indent=4)
            # file.write(model_json)
        print(f"INFO: Model architecture saved to {json_path}")

        return None
    
    def save_model_to_tf(self, model_name):
        DIR_PATH = 'Trained Models'
        if not os.path.exists(DIR_PATH):
            os.makedirs(DIR_PATH)
        self.models[model_name].save(os.path.join(DIR_PATH, model_name), save_format='tf')
        return None
    
    def load_model_from_tf(self, model_name):
        DIR_PATH = 'Trained Models'
        self.models[model_name] = tf.keras.models.load_model(os.path.join(DIR_PATH, model_name))

        if model_name in self.models:
            print(f"INFO: Model '{model_name}' successfully loaded in")

        return None
    
    def get_confusion_matrix(self, model_name, test_dataset):
        """
        Returns the confusion matrix for a given model and test dataset.
        
        Args:
        model: A TensorFlow/Keras model.
        test_dataset: A tf.data.Dataset object, batched into size 32.
        
        Returns:
        A confusion matrix as a NumPy array.
        """
        true_labels = []
        predictions = []

        # print(f"DEBUG: Expected number of samples in test set: {len(list(test_dataset.unbatch()))}")
        
        for batch in test_dataset:
            images, labels = batch

            # -- DEBUG --
            # print("DEBUG: Batch images shape:", images.shape)
            # print("DEBUG: Batch labels shape:", labels.shape)
        
            preds = self.models[model_name].predict(images)
            true_labels.extend(labels.numpy())
            predictions.extend((preds > 0.5).astype(int).flatten())

            # -- DEBUG --
            # print("True labels:", true_labels)
            # print("Predictions:", predictions)
        
        return confusion_matrix(true_labels, predictions)
    
    def test_model_on_directory(self, model_name, path, batch_size=32):

        input_shape = self.models[model_name].input_shape[1:]

        test_set = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            seed=123,
            image_size=(input_shape[0], input_shape[1]),
            batch_size=batch_size,
            label_mode='int',
            shuffle=True
        )

        cm = self.get_confusion_matrix(model_name, test_set)
        tn, fp, fn, tp = cm.ravel()
        print(tn, fp, fn, tp)
        print("Confusion Matrix:\n", cm)

        results = self.models[model_name].evaluate(test_set)
        return print(f"Evaluation results for '{model_name}': {results}")
    

    
    def _count_images_in_classes(self, dataset, class_names):
        class_counts = {class_name: 0 for class_name in class_names}
        for images, labels in dataset.unbatch():
            class_name = class_names[labels.numpy()]
            class_counts[class_name] += 1
        return class_counts
    
    def _create_balanced_dataset(self, dataset, batch_size, seed):

        class_names = dataset.class_names

        class_counts = self._count_images_in_classes(dataset, class_names)
        min_count = min(class_counts.values())

        balanced_dataset = {class_name: [] for class_name in class_names}

        for images, labels in dataset.unbatch():
            class_name = class_names[labels.numpy()]
            if len(balanced_dataset[class_name]) < min_count:
                balanced_dataset[class_name].append((images, labels))
    
    # Convert to tf.data.Dataset
        balanced_images = []
        balanced_labels = []
        for class_name in balanced_dataset:
            for image, label in balanced_dataset[class_name]:
                balanced_images.append(image)
                balanced_labels.append(label)
    
        balanced_images = tf.stack(balanced_images)
        balanced_labels = tf.convert_to_tensor(balanced_labels)
    
        balanced_ds = tf.data.Dataset.from_tensor_slices((balanced_images, balanced_labels))

        balanced_ds = balanced_ds.shuffle(buffer_size=len(balanced_images), seed=seed).batch(batch_size)

        # DEBUG - Count images in each class after balancing
        class_counts_after = self._count_images_in_classes(balanced_ds, class_names)
        print("DEBUG: Class counts after balancing:", class_counts_after)
    
        return balanced_ds
    
    def preprocess(self, image, label):

        # print("DEBUG: Shape of image coming into preprocessing:", image.shape)

        crop_box = [0, 30, 0, 30]  # [start_y, end_y, start_x, end_x]
        image = image / 255.0

        # Define a threshold to identify white pixels
        white_threshold = 100 / 255.0  # Normalized threshold for white pixels

        # Generate random noise with the same shape as the image
        noise = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)

        # Create a mask for white pixels
        white_pixels_mask = tf.reduce_all(image > white_threshold, axis=-1, keepdims=True)

        #  # Replace white pixels with noise
        # image = tf.where(white_pixels_mask, tf.zeros_like(image), image)

        # image = self._crop_image(image, crop_box)
        # image = self._crop_circle(image, (32,32), 22)
        
        return image, label
    
    def random_crop_and_resize(self, image, target_size):
        """
        Randomly crop the image and then resize it to the target size.
        """
        # Randomly crop the image
        cropped_image = tf.image.random_crop(image, size=[tf.shape(image)[0], target_size[0], target_size[1], tf.shape(image)[-1]])
        
        # Resize the cropped image to the target size
        resized_image = tf.image.resize(cropped_image, target_size)
        
        return resized_image
    
    def _crop_image(self, image, crop_box):
        """
        Crop the image to the specified crop_box.
        """
        return image[: ,crop_box[0]:crop_box[1], crop_box[2]:crop_box[3], :]
    
    def _crop_circle(self, image, center, radius):
        """
        Crop the image in a circular region with the given center and radius.
        """
        center_y, center_x = center
        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)

        y, x = np.ogrid[:image.shape[1], :image.shape[2]]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask[distance <= radius] = 1

        mask = tf.convert_to_tensor(mask)
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.tile(mask, [tf.shape(image)[0], 1, 1, 3])

        # Generate a random color
        # random_color = tf.random.uniform(shape=[tf.shape(image)[0], 1, 1, 3], minval=0, maxval=1)

        # # Compute the mean pixel value of the image
        # mean_value = tf.reduce_mean(image, axis=[1, 2], keepdims=True)
        # mean_image = tf.ones_like(image) * mean_value

        # Set the mask region to black (0)
        black_image = tf.zeros_like(image)

        # Create an image filled with the random color
        # random_color_image = tf.ones_like(image) * random_color

        # Generate random noise
        random_noise = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
        return image * mask + random_noise * (1 - mask)

        # return image * mask
        # return image * mask + mean_image * (1 - mask)
        # return image * mask + black_image * (1 - mask)
        return image * mask + random_color_image * (1 - mask)
    
    def _mask_image(self, image, mask_box):
        """
        Mask the image to ignore the specified mask_box.
        """
        mask = np.ones_like(image)
        mask[mask_box[0]:mask_box[1], mask_box[2]:mask_box[3], :] = 0
        image = image * mask
        return image
    
    
    def _load_data_from_directory(self, path, img_height, img_width, batch_size, seed=42, 
                                    flip_binary_labels=False, auto_balance_dataset=False):
        
        if auto_balance_dataset:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                path,
                seed=seed,
                image_size=(img_height, img_width),
                batch_size=1,
                label_mode='int'
            )
            class_names = dataset.class_names
            dataset = self._create_balanced_dataset(dataset, batch_size, seed)

        else:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                path,
                seed=seed,
                image_size=(img_height, img_width),
                batch_size=batch_size,
                label_mode='int',
                shuffle=True
            )
            class_names = dataset.class_names

        if flip_binary_labels:
            dataset = dataset.map(self.flip_binary_labels)

        dataset = dataset.map(self.preprocess)

        # AUTOTUNE = tf.data.AUTOTUNE # Use caching and prefetching to improve performance.
        # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE) # TEMPORARY!!!
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        train_data, val_data, test_data = self._get_dataset_partitions(dataset)

        # -- DEBUG --
        # for images, labels in train_data.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Training Set")
        # for images, labels in val_data.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Validation Set")
        for images, labels in test_data.take(1):
            self.show_batch(images, labels, class_names, dataset_name="Test Set")

        return train_data, val_data, test_data
    
    def flip_binary_labels(self, feature, label):
        return feature, 1 - label
    
    
    def _get_dataset_partitions(self, ds, train_split=0.7, val_split=0.15, test_split=0.15,
                                seed=42, shuffle=False, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=seed)
        
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        test_size = int(test_split * ds_size)
        
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size + val_size).take(test_size)
        
        return train_ds, val_ds, test_ds
    
    def _create_model_from_architecture(self, architecture: str, input_shape):

        if architecture == 'simple_binary_cnn':
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            # model.add(Reshape((-1, 128)))
            # model.add(AttentionLayer())
            # model.add(SpatialAttentionLayer())
            # model.add(CenterBiasedAttentionLayer())
            # model.add(CenterBiasedAttentionLayer(name='center_biased_attention'))
            model.add(Flatten())

            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

        elif architecture == 'chatGPT':
            model = Sequential()
            # First Convolutional Block
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))

            # Second Convolutional Block
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))

            # Third Convolutional Block
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))

            # Fourth Convolutional Block
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))

            # Fifth Convolutional Block
            model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D((2, 2)))
            model.add(Dropout(0.25))

            # Fully Connected Layers
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            # model.add(Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid'))
            model.add(Dense(1, activation='sigmoid'))

            # # Compile the model
            # model.compile(optimizer='adam', 
            #             loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy', 
            #             metrics=['accuracy'])
            
            # return model

        elif architecture == 'ResNet50':
            # Start by defining the input layer with the correct input shape
            input_tensor = Input(shape=(224, 224, 3))
            # Include the Rescaling layer right after the input
            # rescaled_input = Rescaling(scale=1./255)(input_tensor)

            # Apply the correct preprocessing using a Lambda layer
            # processed_input = Lambda(preprocess_input)(input_tensor)
            # processed_input = PreprocessInputLayer()(input_tensor)

            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

            for layer in base_model.layers:
                layer.trainable = False

            # if mods == 'binary':
            # Add custom layers on top of ResNet
            x = base_model.output
            x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
            x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
            predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

            # This is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)

        else:
            raise ValueError(f"'{architecture}' is not a recognised model architecture")

        return model
    
    def train_new_model(self, architecture: str, training_path, auto_balance_dataset, 
                        img_height, img_width, channels=3, batch_size=32,
                        epochs=None, patience=5, seed=42, custom_name=None, save_model=False):
        
        if custom_name is None:
            model_name = f"{img_height}x{img_width}{architecture}"
        else:
            model_name = custom_name
        
        train_data, val_data, test_data = self._load_data_from_directory(
            path=training_path, 
            img_height=img_height, 
            img_width=img_width,
            batch_size=batch_size,
            auto_balance_dataset=auto_balance_dataset,
            seed=seed
        )
        
        model = self._create_model_from_architecture(
            architecture=architecture, 
            input_shape=(img_height, img_width, channels)
        )
        self.models[model_name] = model

        self.models[model_name].compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = []
        if epochs is None:
            epochs = 100
            early_stopping = EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=patience,
                verbose=1,
                mode='min',
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        self.models[model_name].fit(
            train_data, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=val_data, 
            callbacks=callbacks
        )

        cm = self.get_confusion_matrix(model_name, test_data)
        print("Confusion Matrix:\n", cm)

        if save_model:
            self.save_model_to_tf(model_name)

        return None

        
    
    def show_batch(self, image_batch, label_batch, class_names, dataset_name):
        plt.figure(figsize=(16,8), num=f"{dataset_name} Images")
        batch_size = image_batch.shape[0]
        for n in range(min(16, batch_size)):  # Displaying 16 images; adjust this number based on how many images you want to show
            ax = plt.subplot(4, 4, n+1)  # Arrange images in a 4x4 grid
            img = image_batch[n].numpy()
            img = img * 255.0
            # plt.imshow(image_batch[n].numpy().astype("uint8"))  # Convert float type image back to uint8
            plt.imshow(img.astype("uint8"))  # Convert float type image back to uint8
            plt.title(class_names[label_batch[n]])
            plt.axis("off")
        plt.show()

    def _define_preprocessing_layers(self):
        augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        ])

        return augmentation

    def train_model(self, save_name, data_path:str=None)-> None:

        train_ds, val_ds, test_ds = self._load_dataset(data_path)
        # class_count = len(train_ds.class_names)
        augmentation = self._define_preprocessing_layers()

        # AUTOTUNE = tf.data.AUTOTUNE # Use caching and prefetching to improve performance.
        # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model = tf.keras.Sequential([
            augmentation,
            Rescaling(1./255),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

            #Model
        # model = Sequential()
        # model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
        # model.add(MaxPooling2D())

        # model.add(Conv2D(32, (3,3), 1, activation='relu'))
        # model.add(MaxPooling2D())
            
        # model.add(Conv2D(16, (3,3), 1, activation='relu'))
        # model.add(MaxPooling2D())

        # model.add(Flatten())

        # model.add(Dense(256, activation='relu'))
        # #model.add(Dropout(0.5))
        # model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='adam',
                loss='binary_crossentropy',
                #loss = tf.losses.categorical_crossentropy(),
                # loss = 'categorical_crossentropy',
                metrics=['accuracy'])
        
        epochs=10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        # model.evaluate(test_ds)

        # save_name = save_name + '.h5'
        # model.save(save_name)
        self.model = model
        model.save(save_name + '.keras')

        # self.plot_training_history(history)

        return None
    
    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
    
    def load_test_set(self, data_path=None):

        _, _, test_ds = self._load_dataset(data_path)

        return test_ds
    
    def model_predict(self, model_name, image):
        # print("DEBUG: Image shape:", image.shape)
        pred = self.models[model_name].predict(image)
        return pred
    
    def get_explantion(self, model_name, test_image):
        explanation = self.explainer.explain_instance(test_image.astype('float32'), 
                                         classifier_fn=lambda img: self.model_predict(model_name, img), 
                                         top_labels=5, 
                                         hide_color=0, 
                                        #  num_samples=1000,
                                        #  num_samples=2000,
                                         num_samples=5000)  # Increase or decrease depending on the complexity
        
        return explanation
        

    def show_explanation(self, explanation, original_image, truth, predicted_class):
    
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
     

          # Get the image and mask from the explanation
        # image = original_image / 255.0
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        image = temp / 255

        # img_boundry = mark_boundaries(temp, mask)
        img_boundry = mark_boundaries(temp, mask)

        # Create a mask overlay with transparency
        mask_overlay = np.zeros_like(temp)
        mask_overlay[mask == 1] = [0, 1, 0]  # Red color for important regions
        mask_overlay = np.clip(mask_overlay, 0, 1)

        # Blend the original image and the mask overlay
        overlay_image = temp.copy()
        alpha = 0.5  # Transparency factor
        overlay_image[mask == 1] = alpha * temp[mask == 1] + (1 - alpha) * mask_overlay[mask == 1]



        # image = temp / 2 + 0.5

        # plt.imshow(mark_boundaries(temp / 255.0, mask))
        
        # plt.show()

        # The image with the overlay
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        ax[0].imshow(img_boundry)
        plt.title(f"Predicted: , Actual: {truth}")
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Display image with mask
        ax[1].imshow(overlay_image)
        ax[1].set_title('LIME Explanation')
        ax[1].axis('off')

        fig.suptitle(f'Predicted: {predicted_class}, Actual: {truth}', fontsize=16)
      
        self.save_plot(plt)
        plt.show()

        return None

    
    # def show_SHAP_explanation(self, data_path, image, sample_size=100) -> None:
    #     train_ds, _, _ = self._load_dataset(data_path=data_path)

    #     background = self.extract_background(train_ds)

        

    #     # Ensure the model output shape is compatible
    #     def custom_output_wrapper(inputs):
    #         return self.model(inputs)
        
    #     input_layer = tf.keras.Input(shape=(224, 224, 3))
    #     output_layer = self.model(input_layer)

    #     if isinstance(output_layer, tuple):
    #         output_layer = output_layer[0]  # Ensure output is a tensor

    #     wrapped_model = Model(inputs=input_layer, outputs=output_layer)
        

    #     # explainer = shap.DeepExplainer(self.model, background)
    #     # Use DeepExplainer with the wrapped model
    #     explainer = shap.DeepExplainer(wrapped_model, background)

    #     # Ensure the input image is batched correctly
    #     if len(image.shape) == 3:
    #         image = np.expand_dims(image, axis=0)

    #     print(f"Background shape: {background.shape}")
    #     print(f"Image shape: {image.shape}")

    #     shap_values = explainer.shap_values(image)

    #     shap.image_plot(shap_values[0], image)

    #     return None

    # def show_SHAP_explanation(self, data_path, image, sample_size=100) -> None:
    #     train_ds, _, _ = self._load_dataset(data_path=data_path)

    #     background = self.extract_background(train_ds)

    #     print(f"Background shape: {background.shape}")
    #     print(f"Image shape: {image.shape}")

    #     # Wrap the model to ensure proper output handling
    #     input_layer = tf.keras.Input(shape=(224, 224, 3))
    #     output_layer = self.model(input_layer)
        
    #     # Check if the output is a tuple and handle it
    #     if isinstance(output_layer, tuple):
    #         output_layer = output_layer[0]

    #     # Create the wrapped model
    #     wrapped_model = Model(inputs=input_layer, outputs=output_layer)
        
    #     # Print the output layer shape for debugging
    #     print(f"Wrapped model output shape: {wrapped_model.output.shape}")

    #     # Use DeepExplainer with the wrapped model
    #     explainer = shap.DeepExplainer(wrapped_model, background)

    #     # Ensure the input image is batched correctly
    #     if len(image.shape) == 3:
    #         image = np.expand_dims(image, axis=0)

    #     print(f"Batched image shape: {image.shape}")

    #     shap_values = explainer.shap_values(image)

    #     # Since it's a binary classifier, use shap_values[0]
    #     shap.image_plot(shap_values[0], image)

    #     return None

    # def show_SHAP_explanation(self, data_path, image, sample_size=100) -> None:
    #     train_ds, _, _ = self._load_dataset(data_path=data_path)

    #     background = self.extract_background(train_ds)

    #     print(f"Background shape: {background.shape}")
    #     print(f"Image shape: {image.shape}")

    #     # Wrap the model to ensure proper output handling
    #     input_layer = tf.keras.Input(shape=(224, 224, 3))
    #     output_layer = self.model(input_layer)
        
    #     # Check if the output is a tuple and handle it
    #     if isinstance(output_layer, tuple):
    #         output_layer = output_layer[0]

    #     # Create the wrapped model
    #     wrapped_model = Model(inputs=input_layer, outputs=output_layer)
        
    #     # Print the output layer shape for debugging
    #     print(f"Wrapped model output shape: {wrapped_model.output.shape}")

    #     # Ensure the input image is batched correctly
    #     if len(image.shape) == 3:
    #         image = np.expand_dims(image, axis=0)

    #     print(f"Batched image shape: {image.shape}")

    #     # Use DeepExplainer with the wrapped model
    #     explainer = shap.DeepExplainer(wrapped_model, background)

    #     shap_values = explainer.shap_values(image)

    #     # Since it's a binary classifier, use shap_values[0]
    #     shap.image_plot(shap_values[0], image)

    #     return None

    # def show_SHAP_explanation(self, data_path, image, sample_size=100) -> None:
    #     train_ds, _, _ = self._load_dataset(data_path=data_path)

    #     print(train_ds)

        # background = self.extract_background(train_ds)

        # print(f"Background shape: {background.shape}")
        # print(f"Image shape: {image.shape}")

        # # Ensure the input image is batched correctly
        # if len(image.shape) == 3:
        #     image = np.expand_dims(image, axis=0)

        # print(f"Batched image shape: {image.shape}")

        # # Use DeepExplainer with the model directly
        # explainer = shap.DeepExplainer(self.model, background)

        # shap_values = explainer.shap_values(image)

        # # Since it's a binary classifier, use shap_values[0]
        # shap.image_plot(shap_values[0], image)

        # return None
    
    # def extract_background(self, dataset, sample_size=50):
    #     # Prepare a batch iterator
    #     iterator = iter(dataset.unbatch().batch(1).take(sample_size))
    #     # Extract samples and concatenate to form the background dataset
    #     background_data = np.array([batch[0].numpy() for batch, _ in iterator])

    #     print(background_data.shape)

    #     return background_data

    # def extract_background(self, dataset, sample_size=50):
    #     # Prepare a batch iterator
    #     iterator = iter(dataset.unbatch().batch(1).take(sample_size))
    #     # Extract samples and concatenate to form the background dataset
    #     background_data = np.array([batch[0].numpy() for batch in iterator])

    #     background_data = background_data.reshape((sample_size, 224, 224, 3))

    #     print(f"Background data shape: {background_data.shape}")

    #     return background_data

    
    
    def save_plot(self, plot) -> None:
        sub_dir = 'Plots'

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.png"
        path = os.path.join(sub_dir, filename)
        plt.savefig(path)
        

        return None

        # # Create a figure to display
        # plt.figure(figsize=(8, 8))
        # # Show the original image
        # plt.imshow(temp/255)
        # plt.axis('off')
        # # Overlay the mask
        # colored_mask = np.zeros_like(temp/255)
        # for i in range(3):  # Assuming image has three channels (RGB)
        #     colored_mask[:,:,i] = mask * 255  # Assuming mask is 0 where we don't want color
        # plt.imshow(colored_mask, alpha=0.05, cmap='jet')
        # plt.show()

        # image = temp / 255.0
        # binary_mask = mask > 0
        # rgba_image = self.apply_mask(image, binary_mask, color=[255, 0, 0], alpha=0.1)  # Red color, 40% transparency
        # plt.figure(figsize=(8, 8))
        # plt.imshow(rgba_image)
        # plt.axis('off')
        # plt.show()

    # def apply_mask(self, image, mask, color=[255, 0, 0], alpha=0.4):
    #     """
    #     Apply a transparent color mask to an image.
        
    #     Args:
    #     image: Original image (normalized to 0-1 range).
    #     mask: Binary mask where 1 indicates the area to highlight.
    #     color: Color for the overlay as [R, G, B].
    #     alpha: Transparency level of the overlay.
        
    #     Returns:
    #     colored_image: Image with the transparent mask applied.
    #     """
    #     # Create an RGBA version of the original image
    #     colored_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
    #     colored_image[..., :3] = image
    #     colored_image[..., 3] = 1  # Set alpha of original to 1 (no transparency)
        
    #     # Apply color and alpha to the mask
    #     for i in range(3):  # Apply color to RGB channels
    #         colored_image[:, :, i] = np.where(mask, color[i] / 255.0, colored_image[:, :, i])
    #     colored_image[:, :, 3] = np.where(mask, alpha, colored_image[:, :, 3])  # Apply transparency mask
        
    #     return colored_image


    
    # def _preprocess_image(self, image, size: int=256):
    #     """
    #     Apply preprocessing to an image: resizing, normalization, and augmentation.

    #     Args:
    #     image: A decoded image tensor.
    #     size: Resize dimension. Different networks expect different dims.

    #     Returns:
    #     image: The preprocessed image tensor.
    #     """
    #     print("Object type:", type(image))
    #     print("Object content:", image)
    #     image = tf.image.resize(image, [size, size])
    #     image = image / 255.0
    #     image = tf.image.random_flip_left_right(image)
    #     image = tf.image.random_brightness(image, max_delta=0.1)

    #     return image
    
    # def _load_and_preprocess_image(self, path):
    #     """
    #     Load and preprocess an image from a file path.

    #     Args:
    #     path: The file path to the image.

    #     Returns:
    #     image: The preprocessed image tensor.
    #     """
    #     image = tf.io.read_file(path)
    #     image = tf.image.decode_image(image, channels=3)

    #     return  self._preprocess_image(image)

        
    
    # def create_dataset(self, data_path=None, data_set=None, batch_size=32, shuffle_buffer_size=1000):
    #     """
    #     Create a dataset from image files in a directory.

    #     Args:
    #     data_dir: Directory where images are stored, organized by class.
    #     batch_size: The number of images in each batch of data.
    #     shuffle_buffer_size: The buffer size for shuffling the data.

    #     Returns:
    #     dataset: A `tf.data.Dataset` object ready for model training.
    #     """
    #     if (data_path == None and self.data_path == None):
    #         print("ERROR: No path or dataset provided")
    #         return None
    #     elif data_path is not None:
    #         self.data_path = data_path
        
    #     class_names = tf.io.gfile.listdir(self.data_path)
    #     class_names.sort()

    #     class_tensor = tf.constant(class_names)
    #     class_table = tf.lookup.StaticHashTable(
    #         initializer=tf.lookup.KeyValueTensorInitializer(
    #             keys=class_tensor,
    #             values=tf.constant(list(range(len(class_names))), dtype=tf.int64)
    #         ),
    #         default_value=-1
    #     )

    #     list_ds = tf.data.Dataset.list_files(self.data_path + '/*/*', shuffle=False)
    #     list_ds = list_ds.shuffle(len(list_ds), reshuffle_each_iteration=False)
    #     image_ds = list_ds.map(lambda x: self._load_and_preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    #     # label_ds = list_ds.map(lambda x: class_names.index(tf.strings.split(x, '/')[-2]), num_parallel_calls=tf.data.AUTOTUNE)
    #     label_ds = list_ds.map(lambda x: (self._load_and_preprocess_image(x), self.parse_class_name(x, class_table)), num_parallel_calls=tf.data.AUTOTUNE)

    #     dataset = tf.data.Dataset.zip((image_ds, label_ds))
    #     dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    #     dataset = dataset.batch(batch_size)
    #     dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    #     # # DEBUGGING
    #     for images, labels in dataset.take(1):
    #         self._show_batch(images.numpy(), labels.numpy())

    #     return dataset
    
    # @tf.function
    # def parse_class_name(self, file_path, class_table):
    #     parts = tf.strings.split(file_path, '/')
    #     class_name = parts[-2]
    #     return class_table.lookup(class_name)
    
    # def show_batch(self, image_batch, label_batch):
    #     plt.figure(figsize=(10,10))
    #     for n in range(16):
    #         ax = plt.subplot(4,4,n+1)
    #         plt.imshow(image_batch[n])
    #         plt.title(label_batch[n])
    #         plt.axis('off')
    #     plt.show()

    def random_undersample(self, data_dir, target_class, target_size):
        class_dirs = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        for class_dir in class_dirs:
            files = os.listdir(class_dir)
            # if len(files) > target_size and os.path.basename(class_dir) != target_class:
            if len(files) > target_size:
                np.random.shuffle(files)
                files_to_remove = files[target_size:]
                for file in files_to_remove:
                    os.remove(os.path.join(class_dir, file))
                print(f"Removed {len(files_to_remove)} files from {class_dir}")


    def extract_background(self, dataset, sample_size=50):
        # Unbatch the dataset to get individual elements
        dataset = dataset.unbatch().take(sample_size)
        # Extract samples and concatenate to form the background dataset
        background_data = np.array([image.numpy() for image, _ in dataset])

        # print(f"Background data shape: {background_data.shape}")

        return background_data

    def show_SHAP_explanation(self, data_path, image, sample_size=50) -> None:
        train_ds, _, _ = self._load_dataset(data_path)

        background = self.extract_background(train_ds, sample_size)

        # print(f"Background shape: {background.shape}")
        # print(f"Image shape: {image.shape}")

        # Ensure the input image is batched correctly
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # print(f"Batched image shape: {image.shape}")

        # Use DeepExplainer with the model directly
        explainer = shap.DeepExplainer(self.model, background)

        shap_values = explainer.shap_values(image)

        # Since it's a binary classifier, use shap_values[0]
        shap.image_plot(shap_values[0], image)


        return None