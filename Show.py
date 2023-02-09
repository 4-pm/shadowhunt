from skimage.io import imread
from skimage.transform import resize
import glob
import numpy as np
from matplotlib import pyplot as plt
from functions import *
import cv2

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
print('ok')

while (True):
    ret, frame = cam.read()
    output = cv2.resize(frame, (256, 256))
    #cv2.imwrite("newimage.png", output)
    nparray = np.array(output)
    mask = return_mask(nparray)
    mask_of_mask = list(zip(*np.where(mask == 1)))
    if mask_of_mask:
        x, y, x2, y2 = mask_of_mask[0][0], mask_of_mask[0][1], mask_of_mask[-1][0], mask_of_mask[-1][1]
    '''
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 1:
                x, y = j, i
                break
    for i in range(0, len(mask), -1):
        for j in range(0, len(mask[i]), -1):
            if mask[i][j] == 1:
                print(i, j)
        '''
    #frame = cv2.flip(frame, 180)
    frame = cv2.rectangle(frame, (y * 2, x * 2), (y2 * 2, x2 * 2), (255, 0, 0), 3)

    #frame = cv2.resize(frame, (256, 256))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
