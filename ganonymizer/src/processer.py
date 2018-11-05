import cv2
import time
import copy
import numpy as np

from .utils.auxiliary_layer import calc_sml_size, pre_padding, cut_padding, pseudo_mask_division
from .inpaint.glcic.completion import gl_inpaint
from .detection.yolov3.detect import yolo_detecter

class GANonymizer:
    def __init__(self, conf, nms, postproc, large_thresh, prepad_thresh, 
            device, detecter, inpainter, datamean):
        self.conf = conf
        self.nms = nms
        self.device = device
        self.detecter = detecter
        self.inpainter = inpainter
        self.datamean = datamean
        self.postproc = postproc
        self.large_thresh = large_thresh
        self.prepad_thresh = prepad_thresh

        self.time_ssd = 0
        self.time_glcic = 0
        self.time_reconstruct = 0


    def detect(self, input, obj_rec):
        ### detection privacy using SSD
        print('[INFO] Detecting objects related to privacy...')
        begin_ssd = time.time()
        obj_rec = yolo_detecter(input, self.detecter, 
                self.conf, self.nms, obj_rec, self.device)
        elapsed_ssd = time.time() - begin_ssd
        print('[TIME] SSD elapsed time: {:.3f}'.format(elapsed_ssd))
        
        return obj_rec, elapsed_ssd

    
    def create_detected_mask(self, input, mask, obj_rec):
        for rec in obj_rec:
            ul_y, ul_x, dr_y, dr_x = \
                    rec[0], rec[1], rec[0] + rec[2], rec[1] + rec[3]
            cut_img = input[ul_y:dr_y, ul_x:dr_x]
            if cut_img.shape[0] > 1 and cut_img.shape[1] > 1:
                mask[ul_y:dr_y, ul_x:dr_x] = np.ones((cut_img.shape[0], cut_img.shape[1], 3)) * 255
                mask = mask.astype('uint8')

        mask = mask.astype('uint8')
        # inpaint = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
        
        return mask
    

    def reconstruct(self, input, mask, obj_rec):
        ### Inpainting using glcic
        if mask.max() > 0:
            begin_reconst = time.time()
            print('[INFO] Removing the detected objects...')
            self.origin = copy.deepcopy(input).shape

            # prepadding
            is_prepad = {'hu':False, 'hd':False, 'wl':False, 'wr':False}
            input, mask, is_prepad = self.prepadding(input, mask, is_prepad)
            print('check1', input.shape)

            # pseudo mask division
            input, mask = self.PMD(input, mask, obj_rec)
            print('check2', input.shape)

            begin_glcic = time.time()
            output = gl_inpaint(input, mask, self.datamean, \
                    self.inpainter, self.postproc, self.device)
            print('check3', output.shape)
            elapsed_glcic = time.time() - begin_glcic
            print('[TIME] GLCIC elapsed time: {:.3f}'.format(elapsed_glcic))

            # cut prepadding
            print(is_prepad)
            output = self.cutpadding(output, is_prepad)
            print('check4', output.shape)

            elapsed_reconst = time.time() - begin_reconst
            print('[TIME] Reconstruction elapsed time: {:.3f}' \
                    .format(elapsed_reconst))

        else:
            output = input
            elapsed_glcic, elapsed_reconst = 0.0, 0.0

        return output, elapsed_glcic, elapsed_reconst


    def prepadding(self, input, mask, is_prepad):
        ### prepadding
        thresh = self.prepad_thresh
        i, j, k = np.where(mask>=10)
        h, w = input.shape[0], input.shape[1]
        print(h, w)
        print('max, i, j', i.max(), j.max())
        print('min, i, j', i.min(), j.min())
        if  (h - 1) - i.max() < thresh or \
                (w - 1) - j.max() < thresh or \
                i.min() < thresh or j.min() < thresh:
            print('[INFO] Prepadding Processing...')
            input, mask, is_prepad = pre_padding(input, mask, thresh, j, i, input.shape, is_prepad)

        return input, mask, is_prepad


    def cutpadding(self, output, is_prepad):
        ### cut pre_padding
        if is_prepad['hu'] or is_prepad['hd'] or is_prepad['wl'] or is_prepad['wr']:
            output = cut_padding(output, self.origin, is_prepad)

        output = output * 255 # denormalization
        output = output.astype('uint8')

        return output 


    def PMD(self, input, mask, obj_rec):
        ### pseudo mask division
        pmd_f = False
        max = 0
        # old_recs = len(obj_rec)
        # for r in obj_rec:
        #     y, x, h, w = r
        #     if y < 0 or y >= input.shape[0] or \
        #             x < 0 or x >= input.shape[1] or \
        #             h > large_thresh or w > large_thresh:
        #         # h > input.shape[0]*0.8 or w > input.shape[1]/2:
        #         obj_rec.remove(r)
        # print(len(obj_rec) - old_recs)

        for r in obj_rec:
            y, x, h, w = r
            if w > self.large_thresh and h > self.large_thresh:
                pmd_f = True
                square_size = min([w, h])
                if square_size > max:
                    max = square_size
        #         print(r)
        

        if pmd_f:
            print('[INFO] Pseudo Mask Division Processing...')
            h_sml = calc_sml_size(self.large_thresh, input.shape[0], max)
            w_sml = calc_sml_size(self.large_thresh, input.shape[1], max)
            
            input_sml = cv2.resize(input, (w_sml, h_sml))
            mask_sml = cv2.resize(mask, (w_sml, h_sml))
            out_sml = gl_inpaint(input_sml, mask_sml, self.datamean, self.inpainter, self.postproc, self.device)
            out_sml = cv2.resize(out_sml, (input.shape[1], input.shape[0]))
            out_sml = (out_sml * 255).astype('uint8')

            input, mask = pseudo_mask_division(input, out_sml, mask, obj_rec, self.large_thresh)

        return input, mask
