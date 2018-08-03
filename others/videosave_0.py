import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
outfile = 'output.avi'
fps = 30.0
codecs = 'HY264'

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*codecs)
out = cv2.VideoWriter(outfile, fourcc, fps, (frame.shape[0], frame.shape[1]))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
