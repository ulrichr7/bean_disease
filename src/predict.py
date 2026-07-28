import numpy as np
import tensorflow as tf
from preprocessing import preprocess_image

MODEL_PATH = "../models/bean_model.h5"

class_names = [
    "angular_leaf_spot",
    "bean_rust",
    "healthy"
]

model = tf.keras.models.load_model(MODEL_PATH)


def predict(image_path):
    image = preprocess_image(image_path)

    prediction = model.predict(image)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = float(np.max(prediction))

    return predicted_class, confidence