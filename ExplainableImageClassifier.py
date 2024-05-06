import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

class ExplainableImageClassifier:
    def __init__(self, data_path=None, data_set=None, model=None, explainable_method=None):
 
        self.data_path = data_path
        self.model = None
        self.explainer = lime_image.LimeImageExplainer()

        return None
    
    def _load_dataset(self, data_path:str=None):

        if (data_path == None and self.data_path == None):
            raise ValueError("No path to dataset provided")
            
        elif data_path is not None:
            self.data_path = data_path

        img_height = 180
        img_width = 180
        batch_size = 32

        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            self.data_path,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            shuffle=True 
        )

        train_ds, val_ds, test_ds = self.get_dataset_partitions_tf(dataset)

        AUTOTUNE = tf.data.AUTOTUNE # Use caching and prefetching to improve performance.
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
        # train_ds = image_dataset_from_directory(
        # self.data_path,
        # validation_split=0.2,
        # subset="training",
        # seed=123,
        # image_size=(180, 180),
        # batch_size=32)

        # val_ds = image_dataset_from_directory(
        # self.data_path,
        # validation_split=0.2,
        # subset="validation",
        # seed=123,
        # image_size=(180, 180),
        # batch_size=32)

        # DEBUG
        # class_names = train_ds.class_names
        # for images, labels in train_ds.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Training Set")
        # for images, labels in val_ds.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Validation Set")

        return train_ds, val_ds, test_ds
    
    def get_dataset_partitions_tf(self, ds, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1
        
        ds_size = len(ds)
        if shuffle:
            ds = ds.shuffle(shuffle_size, seed=12)
        
        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)
        
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size + val_size)
        
        return train_ds, val_ds, test_ds
    
    def show_batch(self, image_batch, label_batch, class_names, dataset_name):
        plt.figure(figsize=(16,8), num=f"{dataset_name} Images")
        for n in range(16):  # Displaying 16 images; adjust this number based on how many images you want to show
            ax = plt.subplot(4, 4, n+1)  # Arrange images in a 4x4 grid
            plt.imshow(image_batch[n].numpy().astype("uint8"))  # Convert float type image back to uint8
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
            tf.keras.layers.Dense(2, activation='softmax')
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
                loss='sparse_categorical_crossentropy',
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

    def load_model(self, model_name:str) -> None:

        self.model = tf.keras.models.load_model(model_name + '.keras')
        self.model.summary()



        return None
    
    def load_test_set(self, data_path=None):

        _, _, test_ds = self._load_dataset(data_path)

        return test_ds
    
    def model_predict(self, image_batch):
        image_batch = tf.convert_to_tensor(image_batch, dtype=tf.float32) / 255.0
        preds = self.model.predict(image_batch)
        return preds
    
    def get_explantion(self, image):
        explanation = self.explainer.explain_instance(image.astype('double'), 
                                         classifier_fn=self.model_predict, 
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000)  # Increase or decrease depending on the complexity
        
        return explanation
        

    def show_explanation(self, explanation, original_image):
    
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        image = temp /255

        # plt.imshow(mark_boundaries(temp / 255.0, mask))
        # plt.title(f"Top Predicted Class: {explanation.top_labels[0]}")
        # plt.show()

        # The image with the overlay
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        ax[0].imshow(original_image/255)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # Display image with mask
        ax[1].imshow(mark_boundaries(image, mask))
        ax[1].set_title('LIME Explanation')
        ax[1].axis('off')
        plt.show()

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