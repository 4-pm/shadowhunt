import tensorflow as tf


model = tf.keras.models.load_model('./model/')
model.evaluate(tf.keras.utils.load_img("./data/validation_data/class_people/64745.jpg"))