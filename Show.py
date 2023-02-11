from skimage.io import imread
from skimage.transform import resize
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import *
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def return_mask(image):
    #frame = imread(frames[i])
    sample = resize(image, SAMPLE_SIZE)

    predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE)
    predict = np.where(predict < 0.1, 1, 0)
    return predict


inp_layer, out_layer = create_layers()
unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)
unet_like.load_weights('./model/')

# дальше код для визуализации, все для использования в реал тайм выше
# кадры не нужно ресайзить, подаешь фото в виде масcива numpy

cam = cv2.VideoCapture(0)
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)
print('ok')


while (True):
    _, frame = cam.read()
    frame = cv2.resize(frame, (256, 256))
    nparray = np.array(frame)
    mask = return_mask(nparray)
    img = []
    for i in mask:
        gg = []
        for x in i:
            if x == 0:
                gg.append([0, 0, 0])
            else:
                gg.append([255, 255, 255])
        img.append(gg)
    img_to_use = np.array(img).astype(np.uint8)

    img_gray = cv2.cvtColor(img_to_use, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)
    #  contours, hierarchy = cv2.findContours(img_to_use, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #  cv2.drawContours(frame, contours, -1, (255, 0, 0), 3, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
