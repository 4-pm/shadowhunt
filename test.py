from skimage import measure
from skimage.io import imread, imsave, imshow
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter
from os import walk
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import *
import PIL

def return_mask(image):
    #frame = imread(frames[i])
    sample = resize(image, SAMPLE_SIZE)

    predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE)
    predict = np.where(predict < 0.1, 1, 0)
    return predict


inp_layer, out_layer = create_layers()
unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)
unet_like.load_weights('./model/')  # the path to the model

# дальше код для визуализации, все для использовния в реалтайм выше
# кадры не нужно ресайзить, подаешь фото в виде масива numpy


frames = sorted(glob.glob('./data/test/*.jpg'))


for i in range(len(frames)):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), dpi=125)
    frame = resize(imread(frames[i]), SAMPLE_SIZE)
    predict = return_mask(frame)
    ax[0].imshow(frame)
    ax[1].imshow(predict, interpolation='nearest')

    dump = np.copy(frame)
    for i in range(len(dump)):
        for j in range(len(dump[0])):
            dump[i][j] = dump[i][j] * predict[i][j]



    ax[2].imshow(dump)



    plt.show()
    plt.close()






