import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy

image_size = (180, 180)
batch_size = 128



model = tf.keras.models.load_model('./model/model.h5')
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"],)

test_image = tf.keras.preprocessing.image.load_img("./data/validation_data/class_people/1.jpg", target_size=image_size)
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(image_size[0], image_size[1])
result = model.predict(test_image)[0]
print(f"This image is {100 * (1 - result)}% people and {100 * result}% no people.")

test_image = tf.keras.preprocessing.image.load_img("./data/validation_data/class_empty/1.jpg", target_size=image_size)
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(image_size[0], image_size[1])
result = model.predict(test_image)[0]
print(f"This image is {100 * (1 - result)}% people and {100 * result}% no people.")