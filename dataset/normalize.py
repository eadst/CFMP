# -*- coding: utf-8 -*-
# @Date    : April 11, 2021
# @Author  : XD
# @Blog    ：eadst.com


import numpy as np
import cv2
import os


dataset = './data/AID/'
means, stds = [], []
img_list = []
i = 0

for root, dirs, files in os.walk(dataset):
    for img in files:
        img_path = os.path.join(root, img)
        img = cv2.imread(img_path)
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

img_arr = np.concatenate(img_list, axis=3)
img_arr = img_arr.astype(np.float32) / 255.

for i in range(3):
    pixels = img_arr[:, :, i, :].ravel()
    means.append(np.round(np.mean(pixels), 4))
    stds.append(np.round(np.std(pixels), 4))

print("normMean = {}".format(means))
print("normStd = {}".format(stds))

