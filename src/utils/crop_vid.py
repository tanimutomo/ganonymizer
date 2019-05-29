import os
import cv2
import numpy as np

from util import video_writer


def crop_vid(infile, outname, out_fps, start, end):
    cap = cv2.VideoCapture(infile)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))
    writer = video_writer(infile, outname, out_fps, W, H)

    count = 1
    while(cap.isOpened()):
        ret, frame = cap.read()
        if count >= start:
            print('count: {}'.format(count))
            if ret:
                writer.write(frame)
            else:
                break
                raise RuntimeError('Cannot read the video frame')
        if count > end:
            break
        count += 1
    cap.release()
    writer.release()

if __name__ == '__main__':
    infile = 'data/videos/inter10_noon.avi'
    crop_vid(infile, '_cropped', 5.0, 80, 450)

