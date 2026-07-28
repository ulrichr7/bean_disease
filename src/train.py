import tensorflow as tf

def save_model(model):

    model.save("../models/bean_model.h5")

    print("Model saved successfully.")