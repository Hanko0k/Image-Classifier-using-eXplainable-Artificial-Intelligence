import os
from DataManager import ImageManager
from ExplainableImageClassifier import ExplainableImageClassifier


DATA_PATH = os.path.join(os.getcwd(), "Dataset")
ARCHITECTURE = 'SimpleCNN'
IMG_DIMS = (64, 64)
DEFAULT_NAME = f'{IMG_DIMS[0]}x{IMG_DIMS[1]}{ARCHITECTURE}'

classifier = ExplainableImageClassifier()
manager = ImageManager()

classifier.train_new_model(
    architecture=ARCHITECTURE, # See README for up-to-date options
    training_path=DATA_PATH,
    img_dims=IMG_DIMS,
    auto_balance_dataset=True,
#     epochs=20, # 'None' = Early stopping enabled (recommended)
    custom_name=None,
    save_model=True,
    save_history=True,
    data_split=(.7, .15, .15)
)

# classifier.load_model_from_tf(DEFAULT_NAME)

train_ds, _, test_ds = manager.load_data_from_directory(
        path=DATA_PATH,
        data_split=(.7, .15, .15),
        img_dims=IMG_DIMS,
        batch_size=32,
        auto_balance_dataset=True,
)

image = manager.get_rand_image_from_testset(test_set=test_ds, label=0, normalize=True)

# image = manager.image_from_directory_to_nparray(
#         path=os.path.join(os.getcwd(), "Dataset\\01-002-02.bmp"),
#         dims=IMG_DIMS,
#         normalize=True
#         )

explanation = classifier.get_lime_explantion(
        model_name=DEFAULT_NAME, 
        image=image
)

# explanation = classifier.get_superimposed_gradcam(
#     img=image,
#     model_name=DEFAULT_NAME
#     )

# explanation = classifier.get_SHAP_explanation(
#     model_name=DEFAULT_NAME,
#     train_ds=train_ds,
#     image=image,
# )

model_pred = classifier.model_predict(DEFAULT_NAME, image)
classifier.plot_explanation(
        explanation=explanation, 
        original_image=image,
        pred_label=model_pred, 
        ground_truth=0
)