import tensorflow as tf

MODEL_PATH = "../models/bean_model.h5"

def retrain(train_dataset, validation_dataset):

    model = tf.keras.models.load_model(MODEL_PATH)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=5
    )

    model.save(MODEL_PATH)

    return history