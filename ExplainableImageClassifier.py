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
import cv2
from DataManager import ImageManager
from PIL import Image
import pickle
from enum import Enum
import lime.lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import watershed
import pickle

class ExplainableImageClassifier:
    def __init__(self, model=None, explainable_method=None):
        self.image_manager = ImageManager()
        self.models = {}
        self.explainer = lime_image.LimeImageExplainer()
        self.train_data = None
        self.val_data = None
        self.test_data = None

        return None
    
    # def save_architecture(self, model_name):
    #     DIR_PATH = 'Model Architectures'
    #     if not os.path.exists(DIR_PATH):
    #         os.makedirs(DIR_PATH)
            
    #     model_json = self.models[model_name].to_json()
    #     json_path = os.path.join(DIR_PATH, model_name + '.json')
    #     with open(json_path, 'w') as file:
    #         json.dump(json.loads(model_json), file, indent=4)
    #         # file.write(model_json)
    #     print(f"INFO: Model architecture saved to {json_path}")

    #     return None
    
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
    
    def _create_model_from_architecture(self, architecture: str, input_shape):

        if architecture == 'SimpleCNN':
            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            model.add(Flatten())
            model.add(Dense(128, activation='relu'))

            model.add(Dense(1, activation='sigmoid'))

        elif architecture == '5DeepCNN':
            model = Sequential()

            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(256, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
   
            model.add(Conv2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
 
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))

            model.add(Dense(256, activation='relu'))
    
            model.add(Dense(1, activation='sigmoid'))

        elif architecture == 'ResNet50':
            if input_shape[0] != 224 or input_shape[1] != 224:
                raise ValueError("The Resnet architecture requires an input shape of 224 x 224")
            
            input_tensor = Input(shape=(224, 224, input_shape[2]))

            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

            for layer in base_model.layers: # Freeze the pretrained earlier layers to save on training time
                layer.trainable = False

            # Add custom layers on top of ResNet
            x = base_model.output
            x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
            x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
            predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

            model = Model(inputs=base_model.input, outputs=predictions)

        else:
            raise ValueError(f"'{architecture}' is not a recognised model architecture")

        return model
    
    def train_new_model(self, architecture: str, training_path, auto_balance_dataset, img_dims, 
                        channels=3, batch_size=32, epochs=None, patience=5, seed=42, custom_name=None, 
                        save_model=False, save_history=False, data_split=(.7, .15, .15)):
        
        if custom_name is None:
            model_name = f"{img_dims[0]}x{img_dims[1]}{architecture}"
        else:
            model_name = custom_name
        
        train_data, val_data, _ = self.image_manager.load_data_from_directory(
            path=training_path,
            data_split=data_split,
            img_dims=img_dims,
            batch_size=batch_size,
            auto_balance_dataset=auto_balance_dataset,
            seed=seed
        )
        
        self.models[model_name] = self._create_model_from_architecture(
            architecture=architecture, 
            input_shape=(img_dims[0], img_dims[1], channels)
        )

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
        
        history = self.models[model_name].fit(
            train_data, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=val_data, 
            callbacks=callbacks
        )

        if save_model:
            self.save_model_to_tf(model_name)

        if save_history:
            target_dir = os.path.join('Trained Models', model_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            with open(os.path.join(target_dir, f'{model_name}_training_history.pkl'), 'wb') as f:
                pickle.dump(history.history, f)
                                   
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
    
    def model_predict(self, model_name, image):
        pred = self.models[model_name].predict(image)
        return pred
    
    def get_lime_explantion(self, model_name, image):

        explanation = self.explainer.explain_instance(
            image=image, 
            classifier_fn= lambda image: self.model_predict(model_name, image), 
            top_labels=2, 
            num_samples=1000, # Increase or decrease depending on the complexity
            segmentation_fn=SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
            )

        return explanation
    

    def plot_explanation(self, explanation, original_image, truth, predicted_class=None):

        
        if isinstance(explanation, lime.lime_image.ImageExplanation):
            temp, mask = explanation.get_image_and_mask(
                    label=explanation.top_labels[0], 
                    positive_only=True, 
                    negative_only=False,
                    hide_rest=False,
                    num_features=5,
                    min_weight=0.0
                )

            # Create a mask overlay with transparency
            mask_overlay = np.zeros_like(temp)
            mask_overlay[mask == 1] = [0, 1, 0]  # Green color for important regions
            mask_overlay = np.clip(mask_overlay, 0, 1)

            # Blend the original image and the mask overlay
            overlay_image = temp.copy()
            alpha = 0.5  # Transparency factor
            overlay_image[mask == 1] = alpha * temp[mask == 1] + (1 - alpha) * mask_overlay[mask == 1]

            # The image with the overlay
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))

            # Display original image
            ax[0].imshow(original_image)
            plt.title(f"Predicted: , Actual: {truth}")
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            # Display image with mask
            ax[1].imshow(overlay_image)
            ax[1].set_title('LIME Explanation')
            ax[1].axis('off')

            # fig.suptitle(f'Predicted: {predicted_class}, Actual: {truth}', fontsize=16)
            fig.suptitle(f'Predicted: {explanation.top_labels[0]}, Actual: {truth}', fontsize=16)
        
            self.save_plot(plt)
            plt.show()

        return None

    def save_plot(self, plot) -> None:
        sub_dir = 'Plots'

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.png"
        path = os.path.join(sub_dir, filename)
        plt.savefig(path)
        

        return None


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

    def normalize_shap_values(self, shap_values):
       
        shap_min = np.min(shap_values)
        shap_max = np.max(shap_values)

        return (shap_values - shap_min) / (shap_max - shap_min)
        

    def extract_background(self, dataset, sample_size=50):
        # Unbatch the dataset to get individual elements
        dataset = dataset.unbatch().take(sample_size)
        # Extract samples and concatenate to form the background dataset
        background_data = np.array([image.numpy() for image, _ in dataset])

        # print(f"Background data shape: {background_data.shape}")

        return background_data

    def show_SHAP_explanation(self, model_name, train_ds, image, sample_size=100) -> None:
        # train_ds, _, _ = self._load_dataset(data_path)

        background = self.extract_background(train_ds, sample_size)
      
        # Ensure the input image is batched correctly
        # if len(image.shape) == 3:
        #     image = np.expand_dims(image, axis=0)

        explainer = shap.DeepExplainer(self.models[model_name], background)

        shap_values = explainer.shap_values(image)

        shap_values_normalized = self.normalize_shap_values(shap_values)

        # Since it's a binary classifier, use shap_values[0]
        shap.image_plot(shap_values_normalized, image, show=False)

        sub_dir = 'Plots'

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SHAP_plot_{timestamp}.png"
        path = os.path.join(sub_dir, filename)
        plt.savefig(path)

        return None
    
    def make_gradcam_heatmap(self, img_array, model_name, last_conv_layer_name, pred_index=None):
        model = self.models[model_name]

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def superimpose_heatmap(self, img, heatmap, alpha=0.4):
  
        img = img.numpy().astype("float32")
        img = img * 255
        img = img.astype('uint8')
    
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # superimposed_img = heatmap * alpha + img
        superimposed_img = cv2.addWeighted(img, 1, heatmap_rgb, alpha, 0)

        return superimposed_img