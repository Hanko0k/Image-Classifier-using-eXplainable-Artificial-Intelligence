import shap
import numpy as np
import tensorflow as tf

# Verify TensorFlow and Keras versions
print(f"TensorFlow version: {tf.__version__}")

# Step 1: Define a simple model for demonstration
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 2: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Step 3: Create dummy data
background = np.random.rand(50, 224, 224, 3)
image = np.random.rand(1, 224, 224, 3)

# Step 4: Use DeepExplainer with the model directly
explainer = shap.DeepExplainer(model, background)

# Step 5: Generate SHAP values
shap_values = explainer.shap_values(image)

# Step 6: Plot the explanation
shap.image_plot(shap_values, image)