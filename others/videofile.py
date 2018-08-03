import numpy as np
import cv2

interval = 10
video = []

# cap = cv2.VideoCapture('../../data/video/night.avi')
video_file = '../../data/video/night.avi'
cap = cv2.VideoCapture(video_file)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
all_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(all_count, fps, W, H))

count = 0
while(cap.isOpened()):
    if count >= all_count-1:
        print('check1')
    ret, frame = cap.read()
    if count >= all_count-1:
        print('check2')

    count += 1
    if ret:
        if count >= all_count-1:
            print('check3')
        if count % interval == 0:
            if count >= all_count-1:
                print('check4')
            video.append(frame)
            if count >= all_count-1:
                print('check5')
            # cv2.imshow('frame',frame)
            print('count: {}, height:{}, width: {}'.format(count, frame.shape[0], frame.shape[1]))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if count >= all_count-1:
                print('check6')
    else:
        break

print('check7')
cap.release()
cv2.destroyAllWindows()

print('check8')
outfile = '../../data/video/inter{}_{}'.format(interval, video_file.split('/')[-1])
fps = 10
codecs = 'H264'
ret, frames, height, width, ch = np.array([video]).shape
print('check9')

fourcc = cv2.VideoWriter_fourcc(*codecs)
writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
for i in range(frames):
    writer.write(video[i])
writer.release()
