import numpy as np
import argparse
import cv2
import time
import torch
from torch.utils.serialization import load_lua

from inpaint.glcic.inpaint import gl_inpaint
from inpaint.glcic.pre_support import *
from inpaint.glcic.utils import *
from inpaint.glcic.completionnet_places2 import completionnet_places2
from detection.ssd.ssd512 import detect

parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='./images/example_12.jpg')
parser.add_argument('--save_cap_dir', default=None)
parser.add_argument('--video', type=str, default='./data/video/vshort_REC_170511_092456.avi')
parser.add_argument('--prototxt', default='./detection/ssd/cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
parser.add_argument('--model', default='./detection/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained file')
parser.add_argument('--output', default='')
parser.add_argument('--cuda', default=None)
parser.add_argument('--conf', type=float, default=0.15, help='minimum probability to filter weak detections')
parser.add_argument('--fps', type=float, default=10.0)
parser.add_argument('--show', action='store_true')
parser.add_argument('--postproc', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    video_file = args.video
    video = np.array([])
    count_frame = 1
    total_time_ssd = 0
    total_time_glcic = 0
    total_time_inpaint = 0

    if args.cuda != None:
        device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    print('[INFO] device is {}'.format(device))
    print('[INFO] loading video...')
    cap = cv2.VideoCapture(video_file)
    # load our serialized model from dis != Nonek
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))

    t0 = time.time()

    # load Completion Network
    print('[INFO] loading model...')
    model = completionnet_places2
    param = torch.load('./inpaint/glcic/completionnet_places2.pth')
    model.load_state_dict(param)
    model.eval()
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    print('')
    while(cap.isOpened()):
        t1 = time.time()
        ret, frame = cap.read()
        if ret:
            print('[INFO] count frame: {}/{}'.format(count_frame, count))
            
            # detection privacy using SSD
            det_s = time.time()
            obj_rec = []
            mask, obj_rec = detect(frame, net, args.conf, obj_rec)
            det_e = time.time()
            print('[TIME] detection time per frame: {}'.format(det_e - det_s))
            total_time_ssd += det_e - det_s

            # Inpainting using glcic
            if mask.max() > 0:
                inp_s = time.time()
                print('[INFO] Removing specific objects...')
                # pre padding
                origin = frame.shape
                n_input = frame.copy()
                n_mask = mask.copy()
                i, j, k = np.where(n_mask>=10)
                flag = {'hu':False, 'hd':False, 'vl':False, 'vr':False}

                large_thresh = 150
                # rec = detect_large_mask(n_mask)
                
                if i.max() > origin[0] - 5 or j.max() > origin[1] - 5 or i.min() < 4 or j.min() < 4:
                    print('[INFO] prepadding images...')
                    n_input, n_mask, flag = pre_padding(n_input, n_mask, j, i, origin, flag)

                pmd_f = False
                for r in obj_rec:
                    y, x, h, w = r
                    if y < 0 or y >= n_input.shape[0] or \
                        x < 0 or x >= n_input.shape[1] or \
                        h > n_input.shape[0] or w > n_input.shape[1]:
                        obj_rec.remove(r)

                    if w > large_thresh or h > large_thresh:
                        pmd_f = True

                if pmd_f:
                    print('[INFO] pseudo mask division...')
                    input256 = cv2.resize(n_input, (256, 256))
                    mask256 = cv2.resize(n_mask, (256, 256))
                    out256 = gl_inpaint(input256, mask256, datamean, model, args.postproc, device)
                    out256 = cv2.resize(out256, (n_input.shape[1], n_input.shape[0]))
                    out256 = (out256 * 255).astype('uint8')
                    n_input, n_mask = pseudo_mask_division(n_input, out256, n_mask, obj_rec, [256, 256], large_thresh)

                inp_only_s = time.time()
                output = gl_inpaint(n_input, n_mask, datamean, model, args.postproc, device)
                inp_only_e = time.time()
                total_time_glcic += inp_only_e - inp_only_s

                # cut pre_padding
                if flag['hu'] or flag['hd'] or flag['vl'] or flag['vr']:
                    output = cut_padding(output, origin, flag)

                output = output * 255 # innormalization
                output = output.astype('uint8')

                inp_e = time.time()
                print('[TIME] inpainting time per frame: {}'.format(inp_e - inp_s))
                total_time_inpaint += inp_e - inp_s
            else:
                output = frame

            # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            concat = cv2.vconcat([frame, output])
            concat = concat[np.newaxis, :, :, :]
            # print('[CHECK] concat.shape: {}'.format(concat.shape))
            if count_frame == 1:
                video = concat.copy()
            else:
                video = np.concatenate((video, concat), axis=0)
            if args.save_cap_dir != None:
                cv2.imwrite('{}out_{}.png'.format(args.save_cap_dir, count_frame), output)
        else:
            break
        t2 = time.time()
        print('[TIME] elapsed time per frame: {}'.format(t2 - t1))
        print('[TIME] mean time for detecting per frame: {}'.format(total_time_ssd / count_frame))
        print('[TIME] mean time for inpainting per frame: {}'.format(total_time_glcic / count_frame))
        print('[TIME] mean time for whole processing per frame: {}'.format((total_time_ssd + total_time_inpaint) / count_frame))
        count_frame += 1
        print('')
    t3 = time.time()
    print('[TIME] total elapsed time for processing images: {}'.format(t3 - t1))
    cap.release()
    cv2.destroyAllWindows()

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
    t4 = time.time()
    print('[INFO] elapsed time for saving video: {}'.format(t4 - t3))
