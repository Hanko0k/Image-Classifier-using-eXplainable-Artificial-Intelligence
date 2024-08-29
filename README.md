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
  - `seed` (int): An arbitrary number that if held constant will ensure repeatability in how the images are loaded and shuffled. If you later want to load the data back in for a model its important you keep the seed the same as when the model was trained.
  - `flip_binary_labels` (bool): If set true will invert the default labels 0 and 1 that are by default based on the order of subfolders. Useful if '0' is respresting the prescence of something when traditionally 0 indicates a negative result.
  - `auto_balance_dataset` (bool): Having too many (or too few) of a particular class of image can bias the model. Setting this to True will randomly truncate the number of photos in all classes to match the number of photos in the class with the least.
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
        auto_balance_dataset=True)

### `image_from_directory_to_nparray(path, dims, normalize=False)`
- **Description**: Loads an image from directory into memory as a numpy array, which can be used with the models.
- `path` (str): Absolute path of the directory containing the image.
- `dims` (tuple): (height, width) Dimensions of the 2D image. Remember to keep consistent dimensions with what your model expects to see if already trained.
- `normalize` (bool): If set True will normalize the image after loading it into memory. Important to do so if your model was trained on normalized images.
- **Returns**: A 3-channel colour image of size 'dims' as a numpy array.
- **Example**:
  ```python
  image = image_manager.image_from_directory_to_nparray(path=DATA_PATH, (128,128), normalize=True)

### `get_rand_image_from_testset(test_set, label, normalize=False)`
- **Description**: Returns a randomly selected image from a set from a selected class (label).
- `test_set` (_TakeDataset): The set from which to select an image.
- `label` (int): Indicate which class you want the image to come from (0-1 for binary classifiers).
- `normalize` (bool):  If set True will normalize the image after loading it into memory. Important to do so if your model was trained on normalized images.
- **Returns**: A 3-channel colour image as a numpy array.
- **Example**:
  ```python
  image = image_manager.get_rand_image_from_testset(test_set, label=0, normalize=True)

## ExplainableImageClassifier Methods

### `train_new_model(architecture, training_path, auto_balance_dataset, img_dims, channels=3, batch_size=32, epochs=None, patience=5, seed=42, custom_name=None, save_model=False, save_history=False, data_split=(.7, .15, .15))`
- **Description**: Trains a new model that can be saved. This method takes care of a lot of the details of training a model under the hood that don't really need to be manually specified for simple tasks or if youre optimisation isn't a priority - many paarmeters have default values, but can be overwritten.
- `architecture` (str): The name of the architecture to train. Current options available are:
  - 'SimpleCNN': 3 convolutional layers for binary tasks where the tasks doens't require fine detail extraction. Fastest network.
  - '5DeepCNN': 5 convolutional layers for binary tasks with increased density in alter layers.
  - 'ResNet50': A pretrained Residual Network with 50 layers for binary tasks. Pretrained on 'imagenet', base layers are frozen from training and a custom dense layer at the end is trainable. Only works with images with dimensions 224x224. Slowest option but fine detail extration better.
- `training_path` (str): Absolute path of the directory of image files. Expects subfolders to be the subclasses.
- `auto_balance_dataset` (bool): Having too many (or too few) of a particular class of image can bias the model. Setting this to True will randomly truncate the number of photos in all classes to match the number of photos in the class with the least.
- `channels` (int): Default is 3 for the RGB channels of a colour image.
- `batch_size` (int): The amount of images per batch. Larger numbers can speed up training at the cost of memory; diminishing returns for larger numbers. Default=32.
- `epochs` (int): The number of times to show your training data to your model. If this is left as 'None' early stopping will be used where training will stop when validation accuracy doesn't improve anymore. Recommended to keep this as 'None' unless you have a reason not to or training isn't going well.
- `patience` (int): if 'epochs' parameter is given a value, 'pateience' is how many epochs training will continue without seeing significant improvement. Larger numbers can waste time and drag training on, lower numbers risk stopping too early before the network converges. Default=5.
 - `seed` (int): An arbitrary number that if held constant will ensure repeatability in how the images are loaded and shuffled. If you later want to load the data back in for a model its important you keep the seed the same as when the model was trained. Default=42.
- `custom_name` (str): A unique name to call the model. If not specified the name will default to dimensions x architecture. Ex. For the 'SimpleCNN' architecture with an image size of 128x128 the models name will be '128x128SimpleCNN'.
- `save_model` (bool): If set True will save the model into a subdirectory in the project named 'Trained Models'. Default=False.
- `save_history` (bool): If set True will save the training history into a subdirectory named 'Trained Models'. Default=False.
- `data_split` (tuple): How the entire dataset should be split between training, validation and testing sets. Default=(0.7, 0.15, .15) 70% training data, 15% valadation and 15% testing. 
- **Returns**: 'None'. The model belongs to the class. If direct access is needed then use my_explainable_classifier_instance.models[name].
- **Example**:
  ```python
  my_classifier.train_new_model(
        architecture='SimpleCNN',
        training_path=DATA_PATH,
        img_dims=(128,128),
        auto_balance_dataset=True,
        save_model=True,
        data_split=(.7, .15, .15))

### `model_predict(model_name, image)`
- **Description**: Returns the prediction from the model for the provided image.
- `model_name` (str): The name given to the model at training. Default names are image_dims x archiecture Ex. '128x128SimpleCNN'.
- `image` (numpy.array): A 2D image as a numpy array to test the model with and seek an explanation. Image can be obtained by using 'get_rand_image_from_testset' or 'image_from_directory_to_nparray' from the ImageManager class, or some other way. Remember to have an image with the same dimensions as your model was trained with.
- **Returns**: A numpy array. Returns the prediction from the model for the provided image.
- **Example**:
  ```python
  prediction = my_classifier.model_predict('128x128SimpleCNN', image)

### `get_lime_explanation(model_name, image)`
- **Description**: Uses Local Interpretable Model-Agnostic Explanations (LIME) to explain a prediction by a model on a particular image.
- `model_name` (str): The name given to the model at training. Default names are image_dims x archiecture Ex. '128x128SimpleCNN'.
- `image` (numpy.array): A 2D image as a numpy array to test the model with and seek an explanation. Image can be obtained by using 'get_rand_image_from_testset' or 'image_from_directory_to_nparray' from the ImageManager class, or some other way. Remember to have an image with the same dimensions as your model was trained with.
- **Returns**: An ImageExplanation object from LIME. Refer to LIME's documentation for everything this contains. Can be plotted by being passed to 'plot_explanation'.
- **Example**:
  ```python
  explanation = my_classifier.get_lime_explantion(
        model_name='128x128SimpleCNN', 
        image=image_from_testset)

### `get_superimposed_gradcam(img, model_name, alpha=0.4)`
- **Description**: Uses GradCAM to explain a prediction by a model on a particular image. The return is the original image overlayed with a heatmap indicating important parts of the image for the models decision making.
- `model_name` (str): The name given to the model at training. Default names are image_dims x archiecture Ex. '128x128SimpleCNN'.
- `img` (numpy.array): A 2D image as a numpy array to test the model with and seek an explanation. Image can be obtained by using 'get_rand_image_from_testset' or 'image_from_directory_to_nparray' from the ImageManager class, or some other way. Remember to have an image with the same dimensions as your model was trained with.
- `alpha` (float): Adjusts the transparency of the explanation heatmap applied to the original image. Default=0.4.
- **Returns**: A custom 'Explanation' object containing an image with a heatmap overlayed. Can be further plotted by being passed to 'plot_explanation'.
- **Example**:
  ```python
  explanation = my_classifier.get_superimposed_gradcam(
    img=image,
    model_name='128x128SimpleCNN'
    )

### `get_shap_explanation(model_name, train_ds, image)`
- **Description**: Provides an explanation of a the models prediction on a particular image using SHAP values. Requires the training dataset the model was trained with.
- `model_name` (str): The name given to the model at training. Default names are image_dims x archiecture Ex. '128x128SimpleCNN'.
- `train_ds` (_TakeDataset): The set of training images that the model was originally trained with. If not still loaded in, this can be obtained with 'load_data_from_directory' from the ImageManager class and using the same parameters.
- `image` (numpy.array): A 2D image as a numpy array to test the model with and seek an explanation. Image can be obtained by using 'get_rand_image_from_testset' or 'image_from_directory_to_nparray' from the ImageManager class, or some other way. Remember to have an image with the same dimensions as your model was trained with.
- **Returns**: A custom 'Explanation' object containing normalised SHAP values. Can be plotted by being passed to 'plot_explanation'.
- **Example**:
  ```python
  train_ds, _, test_ds = image_manager.load_data_from_directory(
        path=DATA_PATH,
        data_split=(0.7, 0.15, 0.15),
        img_dims=(128,128),
        batch_size=32,
        auto_balance_dataset=True)

  test_image = image_manager.get_rand_image_from_testset(test_set=test_ds, label=0)

  explanation = my_classifier.get_SHAP_explanation(
        model_name='128x128SimpleCNN',
        train_ds=train_ds,
        image=test_image)

### `plot_explanation(explanation, original_image, pred_label=None, ground_truth=None)`
- **Description**: Saves a plot of a provided explanation to a sub directory named 'Plots'.
- `Explanation` (Any): Any of the returned explanations.
- `original_image` (nump.array): The image that was origianally passed to the methods that return an explanation.
- `pred_label` (float): The label (class) the model predicts the original image belongs to. Default=None. If not specified, prediction information wont be on the plot.
- `ground_truth` (int): The ACTUAL label (class) the original image belongs to. Default=None. If not specified, prediction information wont be on the plot.
- **Returns**: 'None'. Shows the plot in a window and saves a copy to a sub directory named 'Plots' as a .png file.
- **Example**:
  ```python
  model_prediction = my_classifier.model_predict('128x128SimpleCNN', image)
  classifier.plot_explanation(
        explanation=explanation, 
        original_image=image,
        pred_label=model_prediction, 
        ground_truth=0
  )