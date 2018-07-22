import cv2
import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--shape', type=list, default=[720, 1280, 3])
# parser.add_argument('--vertical', type=list, required=True)
# parser.add_argument('--horizontal', type=list, required=True)
# parser.add_argument('--output', type=str, required=True)
# args = parser.parse_args()

# shape = args.shape
# v = args.vertical
# h = args.horizontal
# print(shape, v, h)

shape = [720, 1280, 3]
start_y, start_x = 480, 1040
height, width = 240, 240
v = np.array([start_x, start_x + width])
h = np.array([start_y, start_y + height])

move_v = 0
move_h = 0

v = v - move_v
h = h - move_h

mask = np.zeros(shape)
# print(mask[h[0]:h[1], v[0]:v[1], :].shape)
# print((v[1] - v[0], h[1] - h[0], shape[2]).shape)
mask[h[0]:h[1], v[0]:v[1], :] = np.ones((height, width, shape[2])) * 255

cv2.imshow('Mask', mask)
cv2.imwrite('./ex_images/m_v{}-{}_h{}-{}.png'.format(v[0], v[1], h[0], h[1]), mask)
cv2.waitKey(0)
