import os
import cv2
import random
import numpy as np

class CreateRandEdgeMask:
    def __init__(self, rand_mask, mask):
        self.height = rand_mask.shape[0]
        self.width = rand_mask.shape[1]
    
    def edge_sampling(self):
        # 0(top), 1(top-left), 2(left), 3(bottom-left), 4(bottom), 5(bottom-right), 6(right), 7(top-right)
        self.position = random.choise([0, 1, 2, 3, 4, 5, 6, 7])
        self.distance = random.choise([0, 1, 2, 3])
        self.masksize = random.randint(50, 200)

    def large_sampling(self):
        self.masksize = random.randint(120, 400)
        self.position = [
                random.randint(int(self.masksize / 2), self.height - self.masksize),
                random.randint(int(self.masksize / 2), self.width - self.masksize)
                ]

    def calc_center_from_edge(self):
        self.center = [0, 0] # [h, w]

        if self.position in [0, 1, 7]:
            self.center[0] = self.distance + int(self.masksize / 2)
        elif self.position in [3, 4, 5]:
            self.center[0] = self.height - (self.distance + int(self.masksize / 2))

        if self.position in [1, 2, 3]:
            self.center[1] = self.width - (self.distance + int(self.masksize / 2))
        elif self.position in [5, 6, 7]:
            self.center[1] = self.distance + int(self.masksize / 2)

    def calc_center_from_large(self):
        self.center = self.position

    def create_mask(self):
        pass


def video_writer(video, output_name, fps, width, height):
    video_name = video.split('/')[-1]
    outfile = os.path.join(os.getcwd(), 'ganonymizer/data/videos/out{}_{}'.format(output_name, video_name))

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))

    return writer

def concat_all(input, original, output):
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

