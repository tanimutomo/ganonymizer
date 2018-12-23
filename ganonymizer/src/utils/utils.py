import os
import cv2
import random
import numpy as np

class CreateRandMask:
    def __init__(self, height, width):
        self.height = height
        self.width = width
    
    def edge_sampling(self):
        # 0(top), 1(top-left), 2(left), 3(bottom-left), 4(bottom), 5(bottom-right), 6(right), 7(top-right)
        self.position = random.choice([0, 1, 2, 3, 4, 5, 6, 7])
        self.distance = random.choice([0, 1, 2, 3])
        self.masksize = random.randint(50, 200)
        print('--sample--')
        print('position: ', self.position)
        print('distance: ', self.distance)
        print('masksize: ', self.masksize)
        self.calc_center_from_edge()

    def large_sampling(self):
        self.masksize = random.randint(120, 400)
        self.position = [
                random.randint(int(self.masksize / 2), self.height - self.masksize),
                random.randint(int(self.masksize / 2), self.width - self.masksize)
                ]
        self.calc_center_from_large()

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

        print('--center--')
        print('center: ', self.center)

    def calc_center_from_large(self):
        self.center = self.position

    def create_mask(self, rand_mask):
        tl_y = self.center[0] - int(self.masksize / 2)
        tl_x = self.center[1] - int(self.masksize / 2)
        br_y = self.center[0] - int(self.masksize / 2)
        br_x = self.center[1] - int(self.masksize / 2)

        print('--bbox--')
        print('bbox: {}, {}, {}, {}'.format(tl_y, tl_x, br_y, br_x))

        mask_part = rand_mask[tl_y:br_y, tl_x:br_x]
        if mask_part.shape[0] > 0 and mask_part.shape[1] > 0:
            rand_mask[tl_y:br_y, tl_x:br_x] = np.ones((mask_part.shape[0], mask_part.shape[1], 3)) * 255
            # mask = mask.astype('uint8')

        rand_mask = rand_mask.astype('uint8')
        return rand_mask


def check_mask_position(rand_mask, mask):
    assert rand_mask.shape == mask.shape

    print('--rand_mask--')
    print('shape: ', rand_mask.shape)
    print('max: ', np.max(rand_mask))
    print('min: ', np.min(rand_mask))

    if np.max(rand_mask) == 0:
        return False
    sum_masks = rand_mask + mask
    if np.max(sum_masks) == 510:
        return False
    return True


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

