import cv2

img = cv2.imread('../data/images/exp10.jpg')
uly, ulx, rdy, rdx = 50, 50, 450, 450
# uly, ulx, rdy, rdx = 340, 760, 740, 1160
# uly, ulx, rdy, rdx = 56, 312, 456, 712
img_crop = img[uly:rdy, ulx:rdx, :]
cv2.imwrite('../data/images/exp10.png', img_crop)
