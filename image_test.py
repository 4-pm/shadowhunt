import cv2
import numpy as np
from os import walk
import random

def ret_countres(name):
    image = cv2.imread("./data/people/" + name)
    final_wide = 64
    dim = (final_wide, final_wide)

    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    fin_img = image.copy()

    filterd_image = cv2.medianBlur(image, 1)


    imageresult = cv2.cvtColor(filterd_image, cv2.COLOR_BGR2HSV)



    bilateral = cv2.bilateralFilter(imageresult, 15, 75, 75)
    img_grey = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    # get threshold image
    thresh_img = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)

    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create an empty image for contours
    img_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))

    kernel = np.ones((1, 1), np.uint8)
    img_contours = cv2.morphologyEx(img_contours, cv2.MORPH_OPEN, kernel)
    img_contours = cv2.morphologyEx(img_contours, cv2.MORPH_GRADIENT, kernel)

    cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

    image = cv2.resize(img_contours, (500, 500), interpolation=cv2.INTER_AREA)
    return image

files = []

for (dirpath, dirnames, filenames) in walk("./data/people/"):
    files.extend(filenames)
    break

for i in files:
    print(i)
    image = ret_countres(i)

    cv2.imshow(i, image)
    cv2.waitKey(0)
    cv2.destroyWindow(i)
