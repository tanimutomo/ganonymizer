import numpy as np
import argparse
import cv2
import time
import torch
from torch.utils.serialization import load_lua

from inpaint.glcic.inpaint import gl_inpaint
from inpaint.glcic.utils import *
from inpaint.glcic.completionnet_places2 import completionnet_places2
from detection.ssd.ssd512 import detect
from utils.preprocessing import *
from utils.create_mask import *
from utils.design import *


def main():
    args = get_parser()

    # set a device
    device = set_device(args)

    # set networks
    detect_model, inpaint_model, datamean = set_networks(args, device)

    # GANonymizer
    gano = GANonymizer(args, device, detect_model, inpaint_model, datamean)

    if args.video != None:
        apply_to_video(args, gano)
    elif args.image != None:
        apply_to_image(args, gano)
    else:
        print('[ERROR] Not selected an input source.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--image', type=str, help='The input image, when you apply GANonymizer to an image.')
    parser.add_argument('--output', default='')

    parser.add_argument('--conf', type=float, default=0.15, help='minimum probability to filter weak detections')
    parser.add_argument('--large_thresh', type=int, default=120, help='The threshold for PMD processing')
    parser.add_argument('--prepad_thresh', type=int, default=4, help='The threshold for prepadding processing')
    parser.add_argument('--fps', type=float, default=10.0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--postproc', action='store_true')
    parser.add_argument('--cuda', default=None)

    parser.add_argument('--mask', type=str, help='The mask image, when you apply Only reconstruction to an image.')
    parser.add_argument('--manual_mask', type=str, help='The ulx,uly,rdx,rdy of the mask you create')
    parser.add_argument('--center_mask', type=int, help='The size of the mask you create')
    parser.add_argument('--edge_mask', type=str, help='The position(edge/corner),distance(between edges),size of the mask you want to create')
    parser.add_argument('--boxline', type=int, help='Write the bouding box at the reconstruction part')
    parser.add_argument('--save_outframe', default=None)
    parser.add_argument('--save_outimage', type=str, default=None, help='{dir,filename(with extention)} that you want to save output image')

    parser.add_argument('--prototxt', default='./detection/ssd/cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
    parser.add_argument('--model', default='./detection/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained')
    parser.add_argument('--inp_param', default='./inpaint/glcic/completionnet_places2.pth')

    args = parser.parse_args()

    return args


def set_networks(args, device):
    print('[INFO] Loading model...')
    detect_model = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    model = completionnet_places2
    param = torch.load(args.inp_param)
    model.load_state_dict(param)
    model.eval()
    model.to(device)
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    return detect_model, model, datamean


def set_device(args):
    if args.cuda != None:
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print('[INFO] Device is {}'.format(device))
    
    return device


def load_video(args):
    print('[INFO] Loading video...')
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))

    return cap, fps, frames
    

def apply_to_image(args, gano):
    # whole, ssd, glcic, reconst
    elapsed = [0, 0, 0, 0]
    input = cv2.imread(args.image)

    # process
    elapsed, output = process_image(args, input, elapsed, gano)

    if args.save_outimage is not None:
        dir = args.save_outimage.split(',')[0] + '/'
        name = args.save_outimage.split(',')[1] + '.png'
        cv2.imwrite(dir+name, output)
    else:
        name = args.image.split('/')[-1].split('.')[0]
        ext = args.image.split('/')[-1].split('.')[-1]
        save_path = './data/images/{}_out{}.{}'.format(name, args.output, ext)
        cv2.imwrite(save_path, output)


def apply_to_video(args, gano):
    # set variables
    video = np.array([])
    count = 1
    # whole, ssd, glcic, reconst
    elapsed = [0, 0, 0, 0]
    total_time = [0, 0, 0, 0]

    # video data
    cap, origin_fps, frames = load_video(args)

    while(cap.isOpened()):
        print('')
        begin_process = time.time()
        ret, frame = cap.read()
        if ret:
            print('-----------------------------------------------------')
            print('[INFO] Count: {}/{}'.format(count, frames))

            # process
            elapsed, output = process_image(args, frame, elapsed, gano)

            # append frame to video
            video = append_frame(args, frame, output, video, count)

            # print the process info per iteration
            total_time, count = print_info_per_process(args, begin_process, elapsed, count, total_time)

    ### Stop video process
    cap.release()
    cv2.destroyAllWindows()

    ### Save video
    save_video(args, video)


def process_image(args, input, elapsed, gano):
    obj_rec = []

    # detect
    if args.mask is not None:
        mask = args.mask
    elif args.manual_mask is not None:
        mask, obj_rec = create_mask(input.shape, args.manual_mask)
    elif args.center_mask is not None:
        mask, obj_rec = center_mask(input.shape, args.center_mask)
    elif args.edge_mask is not None:
        mask, obj_rec = edge_mask(input.shape, args.edge_mask)
    else:
        obj_rec, elapsed[1] = gano.detect(input, obj_rec)
        mask = np.zeros((input.shape[0], input.shape[1], 3))
        mask = gano.create_detected_mask(input, mask, obj_rec)

    cv2.imwrite('./data/images/mask.png', mask)

    original = input.copy()
    origin_mask = mask.copy()
    if args.boxline is not None:
        boxline = np.zeros((input.shape))
        boxline = create_boxline(mask, obj_rec, boxline, args.boxline)

    # reconstruct
    output, elapsed[2], elapsed[3] = gano.reconstruct(input, mask, obj_rec)

    if args.boxline is not None:
        original = write_boxline(original, origin_mask, boxline)
        output = write_boxline(output, origin_mask, boxline)

    if args.show:
        disp = np.concatenate([original, output, origin_mask], axis=1)
        cv2.imshow('Display', disp)
        cv2.waitKey(0)

    return elapsed, output


def append_frame(args, input, output, video, count):
    ### Append the output frame to the Entire video list
    concat = cv2.vconcat([input, output])
    concat = concat[np.newaxis, :, :, :]
    # print('[CHECK] concat.shape: {}'.format(concat.shape))
    if count == 1:
        video = concat.copy()
    else:
        video = np.concatenate((video, concat), axis=0)
    if args.save_outframe != None:
        cv2.imwrite('{}out_{}.png'.format(args.save_outframe, count), output)

    return video


def print_info_per_process(args, begin, elapsed, count, total):
    ### Print the elapsed time of processing
    elapsed[0] = time.time() - begin
    total[0] += elapsed[0]
    total[1] += elapsed[1]
    total[2] += elapsed[2]
    total[3] += elapsed[3]

    print('[TIME] Whole process time: {:.3f}'.format(elapsed[0]))
    print('-----------------------------------------------------')

    if count % 10 == 0:
        print('')
        print('-----------------------------------------------------')
        print('[INFO] Time Summary')
        print('[TIME] SSD average time per frame: {:.3f}'.format(total[1] / count))
        print('[TIME] GLCIC average time per frame: {:.3f}'.format(total[2] / count))
        print('[TIME] Reconstruction average time per frame: {:.3f}'.format(total[3] / count))
        print('[TIME] Whole process average time per frame: {:.3f}'.format(total[0] / count))
        print('-----------------------------------------------------')

    count += 1

    return total, count


def save_video(args, video):
    video_name = args.video.split('/')[-1]
    outfile = './data/video/out{}_{}'.format(args.output, video_name)
    fps = args.fps
    codecs = 'H264'
    # ret, frames, height, width, ch = video.shape
    frames, height, width, ch = video.shape

    fourcc = cv2.VideoWriter_fourcc(*codecs)
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))

    for i in range(frames):
        writer.write(video[i])
    writer.release()



class GANonymizer:
    def __init__(self, args, device, detect_model, inpaint_model, datamean):
        self.conf = args.conf
        self.device = device
        self.detect_model = detect_model
        self.inpaint_model = inpaint_model
        self.datamean = datamean
        self.postproc = args.postproc
        self.large_thresh = args.large_thresh
        self.prepad_thresh = args.prepad_thresh

        self.time_ssd = 0
        self.time_glcic = 0
        self.time_reconstruct = 0


    def detect(self, input, obj_rec):
        ### detection privacy using SSD
        print('[INFO] Detecting objects related to privacy...')
        begin_ssd = time.time()
        obj_rec = detect(input, self.detect_model, self.conf, obj_rec)
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
            self.origin = input.copy().shape

            # prepadding
            flag = {'hu':False, 'hd':False, 'wl':False, 'wr':False}
            input, mask, flag = self.prepadding(input, mask, flag)

            # pseudo mask division
            input, mask = self.PMD(input, mask, obj_rec)

            begin_glcic = time.time()
            output = gl_inpaint(input, mask, self.datamean, \
                    self.inpaint_model, self.postproc, self.device)
            elapsed_glcic = time.time() - begin_glcic
            print('[TIME] GLCIC elapsed time: {:.3f}'.format(elapsed_glcic))

            # cut prepadding
            output = self.cutpadding(output, flag)

            elapsed_reconst = time.time() - begin_reconst
            print('[TIME] Reconstruction elapsed time: {:.3f}' \
                    .format(elapsed_reconst))

        else:
            output = input
            elapsed_glcic, elapsed_reconst = 0.0, 0.0

        return output, elapsed_glcic, elapsed_reconst


    def prepadding(self, input, mask, flag):
        ### prepadding
        thresh = self.prepad_thresh
        i, j, k = np.where(mask>=10)
        h, w = input.shape[0], input.shape[1]
        if  (h - 1) - i.max() < thresh or \
                (w - 1) - j.max() < thresh or \
                i.min() < thresh or j.min() < thresh:
            print('[INFO] Prepadding Processing...')
            input, mask, flag = pre_padding(input, mask, thresh, j, i, input.shape, flag)

        return input, mask, flag


    def cutpadding(self, output, flag):
        ### cut pre_padding
        if flag['hu'] or flag['hd'] or flag['wl'] or flag['wr']:
            output = cut_padding(output, self.origin, flag)

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
            h_sml = self.calc_sml_size(input.shape[0], max)
            w_sml = self.calc_sml_size(input.shape[1], max)
            
            input_sml = cv2.resize(input, (w_sml, h_sml))
            mask_sml = cv2.resize(mask, (w_sml, h_sml))
            out_sml = gl_inpaint(input_sml, mask_sml, self.datamean, self.inpaint_model, self.postproc, self.device)
            out_sml = cv2.resize(out_sml, (input.shape[1], input.shape[0]))
            out_sml = (out_sml * 255).astype('uint8')

            input, mask = pseudo_mask_division(input, out_sml, mask, obj_rec, self.large_thresh)

        return input, mask

    def calc_sml_size(self, origin, max):
        ratio = self.large_thresh / max
        sml = int(np.floor(origin * ratio))
        sml_do2 = sml % 100
        sml_up = sml - sml_do2
        sml_do2 = (sml_do2 // 4) * 4
        sml = sml_up + sml_do2

        return sml



if __name__ == '__main__':
    main()
