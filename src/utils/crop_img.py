import os
import cv2
import numpy as np


def imcrop(impath, pos, size):
    img = _load_img(impath)
    if pos == 'left':
        out = _left_crop(img, size)
    elif pos == 'center':
        out = _center_crop(img, size)
    cv2.imwrite(impath, out)


def _left_crop(img, size):
    h, w, _ = img.shape
    tl_y = h / 2 - size / 2
    tl_x = w - size
    br_y = h - 1
    br_x = w - 1
    out = _crop_img(img, [tl_y, tl_x, br_y, br_x])
    return out


def _center_crop(img, size):
    h, w, _ = img.shape
    tl_y = h / 2 - size / 2
    tl_x = w / 2 - size / 2
    br_y = tl_y + size
    br_x = tl_x + size
    out = _crop_img(img, [tl_y, tl_x, br_y, br_x])
    return out


def _crop_img(img, rec):
    return img[rec[1]:rec[3], rec[0]:rec[2]]


def _load_img(impath):
    return cv2.imread(impath)

