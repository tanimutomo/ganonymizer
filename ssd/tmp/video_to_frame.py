import cv2
import numpy as np

count = 0
infile_dir = '../../data/video/'
infile = 'REC_170511_092456.avi'
video = []
cap = cv2.VideoCapture('{}{}'.format(infile_dir, infile))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
        # video.append(frame)
        cv2.imwrite('in_{}.png'.format(count), frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    count += 1
cap.release()

# outfile = '{}ex_{}'.format(infile_dir, infile)
fps = 20.0
codecs = 'H264'

print(np.array([video]).shape)
ret, frames, height, width, ch = np.array([video]).shape

fourcc = cv2.VideoWriter_fourcc(*codecs)
writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
for i in range(frames):
    writer.write(video[i])
writer.release()
