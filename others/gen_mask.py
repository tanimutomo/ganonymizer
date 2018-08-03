import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--shape', type=list, default=[720, 1280, 3])
# parser.add_argument('--vertical', type=list, required=True)
# parser.add_argument('--horizontal', type=list, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# shape = args.shape
# v = args.vertical
# h = args.horizontal
# print(shape, v, h)

shape = [720, 1280, 3]
v = [180, 540]
h = [1120, 1280]

mask = np.zeros(shape)
# print(mask[v[0]:v[1], h[0]:h[1], shape[2]].shape)
# print((v[1] - v[0], h[1] - h[0], shape[2]).shape)
mask[v[0]:v[1], h[0]:h[1], :] = np.ones((v[1] - v[0], h[1] - h[0], shape[2])) * 255

cv2.imshow('Mask', mask)
cv2.imwrite(args.output, mask)
cv2.waitKey(0)
