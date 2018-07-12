import numpy as np
import cv2

cap = cv2.VideoCapture('../video/REC_170511_092657.avi')
ret, frame = cap.read()
print(frame.shape)
# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
out = cv2.VideoWriter('out.avi', -1, 20.0, (frame.shape[0],frame.shape[1]))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
