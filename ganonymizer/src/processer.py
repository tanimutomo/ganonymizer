import cv2
import time
import copy
import numpy as np

from .utils.auxiliary_layer import calc_sml_size, pre_padding, cut_padding, pseudo_mask_division
from .inpaint.glcic.completion import gl_inpaint
from .detection.yolov3.detect import yolo_detecter
from .segmentation.deeplabv3.segment import detect_deeplabv3, create_mask, calc_bbox

class GANonymizer:
    def __init__(self, segmentation, conf, nms, postproc, large_thresh,
            prepad_thresh, device, detecter, inpainter, datamean):
        self.segmentation = segmentation
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

    
    def segment(self, input, obj_rec):
        print('[INFO] Detecting objects related to privacy...')
        begin_seg = time.time()
        pred = detect_deeplabv3(input, self.detecter, self.device)
        mask = create_mask(pred)
        obj_rec = calc_bbox(pred, mask, obj_rec)
        elapsed_seg = time.time() - begin_seg
        print('[TIME] DeepLabV3 elapsed time: {:.3f}'.format(elapsed_seg))

        return mask, obj_rec, elapsed_seg


    def detect(self, input, obj_rec, detected_obj):
        ### detection privacy using SSD
        print('[INFO] Detecting objects related to privacy...')
        begin_ssd = time.time()
        obj_rec, detected_obj = yolo_detecter(input, self.detecter, self.conf, self.nms, obj_rec, self.device, detected_obj)
        elapsed_ssd = time.time() - begin_ssd
        print('[TIME] YOLO-V3 elapsed time: {:.3f}'.format(elapsed_ssd))
        
        return obj_rec, elapsed_ssd, detected_obj

    
    def create_detected_mask(self, input, mask, obj_rec):
        for rec in obj_rec:
            tl_y, tl_x, br_y, br_x = \
                    rec[0], rec[1], rec[0] + rec[2], rec[1] + rec[3]
            cut_img = input[tl_y:br_y, tl_x:br_x]
            if cut_img.shape[0] > 1 and cut_img.shape[1] > 1:
                mask[tl_y:br_y, tl_x:br_x] = np.ones((cut_img.shape[0], cut_img.shape[1], 3)) * 255
                mask = mask.astype('uint8')

        mask = mask.astype('uint8')
        # inpaint = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
        
        return mask
    

    def reconstruct(self, input, mask, obj_rec, width_max=None, height_max=None):
        ### Inpainting using glcic
        if mask.max() > 0:
            begin_reconst = time.time()
            print('[INFO] Removing the detected objects...')
            self.origin = copy.deepcopy(input).shape

            if self.segmentation:
                begin_glcic = time.time()
                output = gl_inpaint(input, mask, self.datamean, \
                        self.inpainter, self.postproc, self.device)
                elapsed_glcic = time.time() - begin_glcic
                print('[TIME] GLCIC elapsed time: {:.3f}'.format(elapsed_glcic))
            else:
                # prepadding
                is_prepad = {'hu':False, 'hd':False, 'wl':False, 'wr':False}
                input, mask, is_prepad = self.prepadding(input, mask, is_prepad)

                # pseudo mask division
                input, mask = self.PMD(input, mask, obj_rec, width_max, height_max)

                begin_glcic = time.time()
                output = gl_inpaint(input, mask, self.datamean, \
                        self.inpainter, self.postproc, self.device)
                elapsed_glcic = time.time() - begin_glcic
                print('[TIME] GLCIC elapsed time: {:.3f}'.format(elapsed_glcic))

                # cut prepadding
                output = self.cutpadding(output, is_prepad)

            elapsed_reconst = time.time() - begin_reconst
            print('[TIME] Reconstruction elapsed time: {:.3f}' \
                    .format(elapsed_reconst))

        else:
            output = input
            elapsed_glcic, elapsed_reconst = 0.0, 0.0

        output = output * 255 # denormalization
        output = output.astype('uint8')

        return output, elapsed_glcic, elapsed_reconst


    def prepadding(self, input, mask, is_prepad):
        ### prepadding
        thresh = self.prepad_thresh
        i, j, k = np.where(mask>=10)
        h, w = input.shape[0], input.shape[1]
        if  (h - 1) - i.max() < thresh or \
                (w - 1) - j.max() < thresh or \
                i.min() < thresh or j.min() < thresh:
            print('[INFO] Prepadding Processing...')
            input, mask, is_prepad = pre_padding(input, mask, thresh, j, i, is_prepad)

        return input, mask, is_prepad


    def cutpadding(self, output, is_prepad):
        ### cut pre_padding
        if is_prepad['hu'] or is_prepad['hd'] or is_prepad['wl'] or is_prepad['wr']:
            output = cut_padding(output, self.origin, is_prepad)

        return output 


    def PMD(self, input, mask, obj_rec, width_max, height_max):
        ### pseudo mask division
        is_pmd = False

        for r in obj_rec:
            y, x, h, w = r
            if w > self.large_thresh and h > self.large_thresh:
                is_pmd = True

        if is_pmd:
            print('[INFO] Pseudo Mask Division Processing...')
            h_sml = calc_sml_size(self.large_thresh, input.shape[0], height_max)
            w_sml = calc_sml_size(self.large_thresh, input.shape[1], width_max)
            
            input_sml = cv2.resize(input, (w_sml, h_sml))
            mask_sml = cv2.resize(mask, (w_sml, h_sml))
            out_sml = gl_inpaint(input_sml, mask_sml, self.datamean, self.inpainter, self.postproc, self.device)
            out_sml = cv2.resize(out_sml, (input.shape[1], input.shape[0]))
            out_sml = (out_sml * 255).astype('uint8')

            input, mask = pseudo_mask_division(input, out_sml, mask, obj_rec, self.large_thresh)

        return input, mask
