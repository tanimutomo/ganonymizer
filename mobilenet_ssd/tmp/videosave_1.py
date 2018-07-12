import cv2
import numpy as np

infile = '../video/REC_170511_092657.avi'
video = []
cap = cv2.VideoCapture(infile)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        video.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

outfile = 'out.avi'
fps = 30.0
codecs = 'H264'

print(np.array([video]).shape)
ret, frames, height, width, ch = np.array([video]).shape

fourcc = cv2.VideoWriter_fourcc(*codecs)
writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
for i in range(frames):
    writer.write(video[i])
writer.release()
