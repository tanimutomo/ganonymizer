import numpy as np
import cv2

# cap = cv2.VideoCapture('../video/REC_170511_101428.avi')
# cap = cv2.VideoCapture('../video/REC_170511_101921.avi')
# cap = cv2.VideoCapture('../video/REC_170511_102457.avi')
# cap = cv2.VideoCapture('../video/REC_170511_141559.avi')
# cap = cv2.VideoCapture('../video/REC_170511_091047.avi')
# cap = cv2.VideoCapture('../video/REC_170511_092255.avi') # good
cap = cv2.VideoCapture('../video/REC_170511_092456.avi') # very good

while(cap.isOpened()):
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
