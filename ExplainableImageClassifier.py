import tensorflow as tf
import matplotlib.pyplot as plt

class ExplainableImageClassifier:
    def __init__(self, data_path=None, data_set=None, model=None, explainable_method=None):
        self.models = {}
        self.data_path = data_path

        return None
    
    def _preprocess_image(self, image, size: int=256):
        """
        Apply preprocessing to an image: resizing, normalization, and augmentation.

        Args:
        image: A decoded image tensor.
        size: Resize dimension. Different networks expect different dims.

        Returns:
        image: The preprocessed image tensor.
        """
        print("Object type:", type(image))
        print("Object content:", image)
        image = tf.image.resize(image, [size, size])
        image = image / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)

        return image
    
    def _load_and_preprocess_image(self, path):
        """
        Load and preprocess an image from a file path.

        Args:
        path: The file path to the image.

        Returns:
        image: The preprocessed image tensor.
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)

        return  self._preprocess_image(image)

        
    
    def create_dataset(self, data_path=None, data_set=None, batch_size=32, shuffle_buffer_size=1000):
        """
        Create a dataset from image files in a directory.

        Args:
        data_dir: Directory where images are stored, organized by class.
        batch_size: The number of images in each batch of data.
        shuffle_buffer_size: The buffer size for shuffling the data.

        Returns:
        dataset: A `tf.data.Dataset` object ready for model training.
        """
        if (data_path == None and self.data_path == None):
            print("ERROR: No path or dataset provided")
            return None
        elif data_path is not None:
            self.data_path = data_path
        
        class_names = tf.io.gfile.listdir(self.data_path)
        class_names.sort()

        class_tensor = tf.constant(class_names)
        class_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=class_tensor,
                values=tf.constant(list(range(len(class_names))), dtype=tf.int64)
            ),
            default_value=-1
        )

        list_ds = tf.data.Dataset.list_files(self.data_path + '/*/*', shuffle=False)
        list_ds = list_ds.shuffle(len(list_ds), reshuffle_each_iteration=False)
        image_ds = list_ds.map(lambda x: self._load_and_preprocess_image(x), num_parallel_calls=tf.data.AUTOTUNE)
        # label_ds = list_ds.map(lambda x: class_names.index(tf.strings.split(x, '/')[-2]), num_parallel_calls=tf.data.AUTOTUNE)
        label_ds = list_ds.map(lambda x: (self._load_and_preprocess_image(x), self.parse_class_name(x, class_table)), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.zip((image_ds, label_ds))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # # DEBUGGING
        for images, labels in dataset.take(1):
            self._show_batch(images.numpy(), labels.numpy())

        return dataset
    
    @tf.function
    def parse_class_name(self, file_path, class_table):
        parts = tf.strings.split(file_path, '/')
        class_name = parts[-2]
        return class_table.lookup(class_name)
    
    def _show_batch(self, image_batch, label_batch):
        plt.figure(figsize=(10,10))
        for n in range(16):
            ax = plt.subplot(4,4,n+1)
            plt.imshow(image_batch[n])
            plt.title(label_batch[n])
            plt.axis('off')
        plt.show()