# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.callbacks import TensorBoard
# import datetime
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
from ExplainableImageClassifier import ExplainableImageClassifier


path = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset"
classifier = ExplainableImageClassifier()
classifier.create_dataset(data_path=path)