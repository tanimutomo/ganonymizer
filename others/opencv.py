import numpy as np
import cv2

img = cv2.imread('../images/example_12.jpg')
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
rect = np.array([10, 30, 100, 150])
print('img: ', img.shape)
print('res: ', res.shape)
# 長方形の描画
cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (255, 255, 255), 3)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# rectの部分だけを取ってきている．
cut_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
print('cut_img1: ', cut_img.shape)
cv2.imshow('cut_img1', cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 取ってきたcut_imgを20分の一のサイズにしている．
cut_img = cv2.resize(cut_img,(rect[2]//20, rect[3]//20))
print('cut_img2: ', cut_img.shape)
cv2.imshow('cut_img2', cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 20分の一のサイズにしたものを元のcut_imgのサイズに無理やり広げる．
cut_img = cv2.resize(cut_img,(rect[2], rect[3]),cv2.INTER_NEAREST)
print('cut_img3: ', cut_img.shape)
cv2.imshow('cut_img3', cut_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 無理やり広げたものを画像に戻すと，モザイクとなる.
img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]=cut_img
print('img: ', img.shape)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

