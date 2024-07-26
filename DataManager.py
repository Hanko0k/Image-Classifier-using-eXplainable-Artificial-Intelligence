import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np


class ImageManager:
    def __init__(self):

        return None

    def load_data_from_directory(self, path, data_split, img_dims, batch_size, seed=42, 
                                flip_binary_labels=False, auto_balance_dataset=False):
            
        if auto_balance_dataset:
            dataset = image_dataset_from_directory(
                path,
                seed=seed,
                image_size=img_dims,
                batch_size=1,
                label_mode='int'
            )
            class_names = dataset.class_names
            dataset = self._create_balanced_dataset(dataset, batch_size, seed)

        else:
            dataset = image_dataset_from_directory(
                path,
                seed=seed,
                image_size=img_dims,
                batch_size=batch_size,
                label_mode='int',
                shuffle=True
            )
            class_names = dataset.class_names

        if flip_binary_labels:
            dataset = dataset.map(self._flip_binary_labels)

        dataset = dataset.map(self._preprocess)

        # AUTOTUNE = tf.data.AUTOTUNE # Use caching and prefetching to improve performance.
        # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        train_data, val_data, test_data = self._get_dataset_partitions(dataset, data_split)

        # -- DEBUG --
        # for images, labels in train_data.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Training Set")
        # for images, labels in val_data.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Validation Set")
        # for images, labels in test_data.take(1):
        #     self.show_batch(images, labels, class_names, dataset_name="Test Set")

        return train_data, val_data, test_data
        
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
    
    def _count_images_in_classes(self, dataset, class_names):
        class_counts = {class_name: 0 for class_name in class_names}
        for _, labels in dataset.unbatch():
            class_name = class_names[labels.numpy()]
            class_counts[class_name] += 1
        return class_counts
    
    def _get_dataset_partitions(self, ds, data_split, seed=42, shuffle=False, shuffle_size=10000):
        train_split = data_split[0]
        val_split = data_split[1]
        test_split = data_split[2]

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
    
    def _preprocess(self, image, label):
        image = image / 255.0 # Normalise

        # Create a mask for white pixels
        white_pixels_mask = tf.reduce_all(image > white_threshold, axis=-1, keepdims=True)

        #  # Replace white pixels with noise
        # image = tf.where(white_pixels_mask, tf.zeros_like(image), image)
        
        return image, label
    
    def _random_crop_and_resize(self, image, target_size):
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

        Parameters:
        -----------
        image : Any
            2D image to be cropped.
        crop_box : list
            Provided in the format [start_y, end_y, start_x, end_x].
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
    
    def _flip_binary_labels(self, feature, label):
        return feature, 1 - label