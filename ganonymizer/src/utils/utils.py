import os
import cv2
import numpy as np

def video_writer(video, output_name, fps, width, height):
    video_name = video.split('/')[-1]
    outfile = os.path.join(os.getcwd(), 'ganonymizer/data/videos/out{}_{}'.format(output_name, video_name))

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))

    return writer

def concat_inout(input, original, output):
    output_f2 = cv2.resize(output, None, fx=2, fy=2)
    concat = np.concatenate([input, original], axis=0)
    concat = np.concatenate([concat, output_f2], axis=1)
    return concat

def adjust_imsize(input):
    height = input.shape[0]
    width = input.shape[1]
    new_h = height - (height % 4)
    new_w = width - (width % 4)
    input = cv2.resize(input, (new_w, new_h))

    return input

def load_video(video):
    print('[INFO] Loading video...')
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))

    return cap, fps, frames, W, H

def extend_rec(obj_rec, input):
    new_recs = []
    for rec in obj_rec:
        sy, sx, h, w = rec
        sy = sy - int(h * 0.05)
        sx = sx - int(w * 0.05)
        h = h + int(h * 0.10)
        w = w + int(w * 0.05)

        if sy < 0:
            sy = 0
        if sx < 0:
            sx = 0
        if sy + h >= input.shape[0]:
            h = input.shape[0] - sy - 1
        if sx + w >= input.shape[1]:
            w = input.shape[1] - sx - 1

        new_recs.append([sy, sx, h, w])

    return new_recs

