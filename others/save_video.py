import cv2
import numpy as np

infile = '../data/video/in_soccer_2.avi'
video = []

cap = cv2.VideoCapture(infile)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))

count = 1
n_w, n_h = int(W / 4), int(H / 4)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if count >= 300 and count <= 600:
            frame = cv2.resize(frame, (n_w, n_h))
            # cv2.imshow('frame', frame)
            video.append(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    else:
        break
    count += 1
    if count >= 600:
        break
cap.release()

outfile = '../data/video/ex_small_{}'.format(infile.split('/')[-1])
fps = fps
codecs = 'H264'

print(np.array([video]).shape)
ret, frames, height, width, ch = np.array([video]).shape

fourcc = cv2.VideoWriter_fourcc(*codecs)
writer = cv2.VideoWriter(outfile, fourcc, fps, (n_w, n_h))
for i in range(frames):
    writer.write(video[i])
writer.release()
