from ExplainableImageClassifier import ExplainableImageClassifier
import random


path = "C:\\Users\\David\\Documents\\Image-Classifier-using-eXplainable-Artificial-Intelligence\\braeburn binary classifier dataset"
classifier = ExplainableImageClassifier()
# classifier.train_model(data_path=path, save_name="simple_cnn")

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




classifier.load_model("simple_cnn")
test_ds = classifier.load_test_set(data_path=path)

images = []
for img_batch, label_batch in test_ds:
    for img, label in zip(img_batch, label_batch):
        if label == 0:
            images.append(img.numpy().astype("uint8"))

test_image = random.choice(images)

explanation = classifier.get_explantion(test_image)
classifier.show_explanation(explanation, original_image=test_image)
