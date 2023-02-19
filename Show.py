from skimage.transform import resize
import numpy as np
from functions import *
import cv2


def return_mask(image):
    sample = resize(image, SAMPLE_SIZE)
    predict = unet_like.predict(sample.reshape((1,) + SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE)
    predict = np.where(predict < 0.1, 1, 0)
    return predict


inp_layer, out_layer = create_layers()
unet_like = tf.keras.Model(inputs=inp_layer, outputs=out_layer)
unet_like.load_weights('./model/')

#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('3.mp4')
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)
print('ok')

while True:
    _, frame = cam.read()
    frame = cv2.resize(frame, (640, 450))
    nparray = np.array(frame)
    mask = return_mask(nparray)
    img_to_use = np.zeros(mask.shape + (3,), dtype=np.uint8)
    img_to_use[mask > 0] = (255, 255, 255)
    img_to_use = cv2.resize(img_to_use, (640, 450))
    img_gray = cv2.cvtColor(img_to_use, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    count_people = 0
    for i in contours:
        if cv2.contourArea(i) > 1000:
            count_people += 1
            cv2.drawContours(image=frame, contours=i, contourIdx=-1, color=(0, 255, 0),
                     thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=frame, text=str(count_people), org=(50, 100), fontScale=5, fontFace=cv2.FONT_HERSHEY_PLAIN,
                color=(255, 0, 0), thickness=5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
