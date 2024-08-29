# Image-Classifier-using-eXplainable-Artificial-Intelligence

## Description

An eXplainable AI framework that provides an abstraction layer for TensorFlow that allows the training of models as well as encapsulating AI explanation tools LIME, GradCAM and SHAP. The motivation was to provide and all-in-one solution from training models through to explaining their outputs.

The project only currently supports binary classification tasks, but additonal architectures could be added in the future to support other tasks.

## Getting started

### Dependencies

* lime==0.2.0.1
* matplotlib==3.9.0
* numpy==1.26.4
* opencv_python==4.10.0.82
* Pillow==10.3.0
* scikit_learn==1.5.0
* shap==0.45.1
* tensorflow==2.15.1
* tensorflow_intel==2.15.1

## Version History

* 0.1.0
  * Initial Release

## How to Use

* Install the above dependencies. Versions are quite important as the explanation methods don't always work with the current version of TensorFlow.
* Import into your project 'ImageManager' from DataManager.py and 'ExplainableImageClassifier' from ExplainableImageClassifier.py.
* Make use of the below methods to train and use the models.

## ImageManager Methods

### `load_data_from_directory(path, data_split, img_dims, batch_size, seed=42, flip_binary_labels=False, auto_balance_dataset=False, preprocess=True)`
- **Description**: Loads image data in from directory into training, validation and test datasets based on 'data_split'. Given 'seed' remains constant the images will always load and shuffle the same way.
- **Parameters**:
  - `path` (str): Absolute path of the directory of image files. Expects subfolders to be the subclasses.
  - `data_split` (tuple): Three float values representing the % split for training, validation and test sets (train, val, test). Values must sum to 1.0.
  - `img_dims` (tuple): (height, width) Dimensions of the 2D image data. Remember to keep consistent dimensions with what your model expects to see if already trained.
  - `batch_size` (int): Size of the batches of images. Larger batch sizes generally speed up training.
  - `seed` (int): An arbitrary number that if held constant will ensure repeatability in how the images are loaded and shuffled. If you later want to load the data back in                    for a model its important you keep the seed the same as when the model was trained.
  - `flip_binary_labels` (bool): If set true will invert the default labels 0 and 1 that are by default based on the order of subfolders. Useful if '0' is respresting the                                    prescence of something when traditionally 0 indicates a negative result.
  - `auto_balance_dataset` (bool): Having too many (or too few) of a particular class of image can bias the model. Setting this to True will randomly truncate the number of                                    photos in all classes to match the number of photos in the class with the least.
  - `preprocess` (bool): Applies general preprocessing to images such as normalization.
  - **Returns**:
    - `train_data`: Batches of images representing the training data.
    - `val_data` Batches of images representing the valadation data.
    - `test_data`: Batches of images representing the testing data.
- **Example**:
  ```python
  train_ds, val_ds, test_ds = image_manager.load_data_from_directory(
        path=DATA_PATH,
        data_split=(0.7, 0.15, 0.15),
        img_dims=(128,128),
        batch_size=32,
        auto_balance_dataset=True,
        preprocess=False)
```
### `image_from_directory_to_nparray(path, dims, normalize=False)`
- `path` (str): Absolute path of the directory containing the image.
- `dims` (tuple): (height, width) Dimensions of the 2D image. Remember to keep consistent dimensions with what your model expects to see if already trained.
- `normalize` (bool): If set True will normalize the image after loading it into memory. Important to do so if your model was trained on normalized images.
- **Returns**: A 3-channel colour image of size 'dims' as a numpy array.

### `get_rand_image_from_testset(test_set, label, normalize=False)
- `test_set` (_TakeDataset): The set from which to select an image.
- `label` (int): Indicate which class you want the image to come from (0-1 for binary classifiers).
- `normalize` (bool):  If set True will normalize the image after loading it into memory. Important to do so if your model was trained on normalized images.
- **Returns**: A 3-channel colour image as a numpy array.

### `is_batched(obj)`
-`obj` (_TakeDataset): The object representing the set of images being tested for whther it is batched or not.
- **Returns**: True or False