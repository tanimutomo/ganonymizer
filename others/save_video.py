import cv2
import numpy as np

infile = '../data/video/inter10_noon.avi'
video = []

cap = cv2.VideoCapture(infile)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))

count = 1
n_w, n_h = int(W / 3), int(H / 3)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if count >= 1 and count <= 1500:
            frame = cv2.resize(frame, (n_w, n_h))
            # cv2.imshow('frame', frame)
            video.append(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    else:
        break
    count += 1
    if count >= 1500:
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
