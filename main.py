from ExplainableImageClassifier import ExplainableImageClassifier
import random
import numpy as np
import shap
from tensorflow.keras.preprocessing import image

TRAIN_PATH = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\Datasets\\Augmented Mixed Binary"
# ANOTHER_PATH = "C:\\Users\\David\\Documents\\granny smith binary classifier dataset"
# STEPHS_PATH = "C:\\Users\\David\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\stephs pics"

MODEL_NAME = 'chatGPT'
IMG_HEIGHT = 128
IMG_WIDTH = 128
DIMS = str(IMG_HEIGHT)+'x'+str(IMG_WIDTH)

classifier = ExplainableImageClassifier()

# classifier.train_new_model(
#     architecture=MODEL_NAME,
#     training_path=TRAIN_PATH,
#     img_height=IMG_HEIGHT, 
#     img_width=IMG_WIDTH,
#     auto_balance_dataset=True,
#     # epochs=20,
#     patience=10,
#     custom_name = None,
#     save_model = True
# )

classifier.load_model_from_tf(DIMS+MODEL_NAME)

train_ds, _, test_ds = classifier._load_data_from_directory(
                    path=TRAIN_PATH,
                    img_height=IMG_HEIGHT,
                    img_width=IMG_WIDTH,
                    batch_size=32,
                    auto_balance_dataset=True
)

defective_samples = []
for img_batch, label_batch in test_ds:
    for img, label in zip(img_batch, label_batch):
        if label == 0:
            # images.append(img)
            # labels.append(label)
            defective_samples.append((img, label))

# test_ds = test_ds.unbatch()
# samples = list(test_ds)
# defective_samples = []
# for img, label in samples:
#     if label == 0:
#         defective_samples.append((img, label))

rand_sample = random.choice(defective_samples)
img, label = rand_sample
show_image = img.numpy().astype("float32")
show_image = show_image * 255
show_image = show_image.astype("uint8")

# test_image = img.numpy().astype("float32")

fake_img_batch = np.expand_dims(img, axis=0)
# classifier.visualize_attention_map('64x64simple_binary_cnn', test_image)
predicted_label = classifier.models[DIMS+MODEL_NAME].predict(fake_img_batch)

explanation = classifier.get_explantion(model_name=DIMS+MODEL_NAME, test_image=img)
classifier.show_explanation(explanation, original_image=show_image, truth=label, predicted_class=predicted_label)

# # Create an explainer object
# explainer = shap.GradientExplainer(classifier.models[DIMS+MODEL_NAME], fake_img_batch)
# shap_values = explainer.shap_values(fake_img_batch)

# # Visualize the SHAP values
# shap.image_plot(shap_values, fake_img_batch)

# classifier.show_SHAP_explanation(
#     model_name=DIMS+MODEL_NAME, 
#     train_ds=train_ds,
#     image=fake_img_batch,
# )










# classifier.load_pretrained_model(model='ResNet50', mods='binary')
# classifier.train_loaded_model()

# classifier.train_model(data_path=path, save_name="simple_cnn_2")
# test_loss, test_accuracy = classifier.model.evaluate(test_ds)
# print("Test Accuracy:", test_accuracy)
# print("Test Loss:", test_loss)

# Take one batch and one image from the dataset
# for images, labels in test_ds.take(1):
#     test_image = images[0].numpy()
#     test_label = labels[0].numpy()
#     break  # Only take the first image for demonstration

# target_class = 'defective'
# target_index = test_ds.class_names.index(target_class)
# target_index = 0

# for images, labels in test_ds:
#         for i in range(len(labels)):
#             if labels[i] == target_index:
#                 test_image = images[0].numpy()
#                 test_label = labels[0].numpy()
#                 break  # Only take the first image for demonstration

# classifier.load_model("ResNet50_2")

# path = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset"
# classifier.random_undersample(data_dir=path, target_class="defective", target_size=563)
# test_ds = classifier.load_test_set(data_path=path)




# cm = classifier.get_confusion_matrix(test_ds)
# print("Confusion Matrix:\n", cm)



# ground_truth = {
#     'no defect': 0,
#     'defective': 1
# }

# images = []
# for img_batch, label_batch in test_ds:
#     for img, label in zip(img_batch, label_batch):
#         if label == ground_truth['no defect']:
#             images.append(img)

# image = random.choice(images)
# show_image = image.numpy().astype("uint8")
# test_image = image.numpy().astype("float32")

# classifier.show_SHAP_explanation(data_path=path, image=test_image, sample_size=50)





# Predict the class using the model



# image_batch = np.expand_dims(test_image, axis=0)  # Model expects a batch of images

# prediction_probs = classifier.model.predict(image_batch)
# print("DEBUG: Raw prediction probabilities:", prediction_probs)
# predicted_class = (prediction_probs > 0.5).astype(int)
# print(f"DEBUG: Predicted class is {predicted_class}")
# if predicted_class == 1:
#     predicted_class = 'Defective'
# else:
#     predicted_class = 'No defect'


# explanation = classifier.get_explantion(test_image)
# classifier.show_explanation(explanation, original_image=show_image, truth='No Defect', predicted_class=predicted_class)