import os
import cv2
import numpy as np


def imcrop(impath, pos, size):
    img = _load_img(impath)
    if pos == 'left':
        out = _left_crop(img, size)
    elif pos == 'center':
        out = _center_crop(img, size)
    elif pos == 'bottom_left':
        out = _bottom_left_crop(img, size)
    cv2.imwrite(impath, out)


def _left_crop(img, size):
    h, w, _ = img.shape
    tl_y = int(h / 2 - size / 2)
    tl_x = w - size
    br_y = tl_y + size
    br_x = w
    out = _crop_img(img, [tl_y, tl_x, br_y, br_x])
    return out


def _center_crop(img, size):
    h, w, _ = img.shape
    tl_y = int(h / 2 - size / 2)
    tl_x = int(w / 2 - size / 2)
    br_y = tl_y + size
    br_x = tl_x + size
    out = _crop_img(img, [tl_y, tl_x, br_y, br_x])
    return out


def _bottom_left_crop(img, size):
    h, w, _ = img.shape
    tl_y = h - size
    tl_x = w - size
    br_y = h
    br_x = w
    out = _crop_img(img, [tl_y, tl_x, br_y, br_x])
    return out


def _crop_img(img, rec):
    return img[rec[0]:rec[2], rec[1]:rec[3], :]


def _load_img(impath):
    return cv2.imread(impath)


