import tensorflow as tf

IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    return image