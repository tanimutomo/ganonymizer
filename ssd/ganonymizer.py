import numpy as np
import argparse
import cv2
import time
import torch
from torch.utils.serialization import load_lua

from glcic.inpaint import gl_inpaint
from glcic.pre_support import *
from glcic.utils import *
from glcic.completionnet_places2 import completionnet_places2


def detect_person(image, datamean, model, postproc):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (512, 512)), 1.0, (512, 512), 127.5)

    # print('[INFO] computing object detections...')
    net.setInput(blob)
    detections = net.forward()
    mask = np.zeros((image.shape[0], image.shape[1], 3))

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence(i.e., probability) associated with prediciton
        idx = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if idx == 2 or idx == 6 or idx == 7 or idx == 15 or idx == 14:
            # only person, car, bicycle, motorbike, bus
            if confidence > args.confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')
                label = '{}: {:2f}%'.format(CLASSES[idx], confidence * 100)
                print('[DETECT] {}'.format(label))
                
                # mosaic
                cut_img1 = image[start_y:end_y, start_x:end_x]
                if cut_img1.shape[0] > 1 and cut_img1.shape[1] > 1:
                    mask[start_y:end_y, start_x:end_x] = np.ones((cut_img1.shape[0], cut_img1.shape[1], 3)) * 255
                    mask = mask.astype('uint8')

    mask = mask.astype('uint8')
    # inpaint = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
    return mask

parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='./images/example_12.jpg')
parser.add_argument('--save_cap_dir', type=str, default='../data/video/frames3/')
parser.add_argument('--video', type=str, default='../data/video/vshort_REC_170511_092456.avi')
parser.add_argument('--prototxt', default='./cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
parser.add_argument('--model', default='./weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained file')
parser.add_argument('--output', default='')
parser.add_argument('--cuda', default='0')
parser.add_argument('--confidence', type=float, default=0.15, help='minimum probability to filter weak detections')
parser.add_argument('--fps', type=float, default=10.0)
parser.add_argument('--show', action='store_true')
parser.add_argument('--postproc', action='store_true')

# initialize the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


if __name__ == '__main__':
    args = parser.parse_args()
    video_file = args.video
    video = []
    count_frame = 1
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('[INFO] loading video...')
    cap = cv2.VideoCapture(video_file)
    # load our serialized model from disk
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
    param = torch.load('./glcic/completionnet_places2.pth')
    model.load_state_dict(param)
    model.eval()
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    while(cap.isOpened()):
        t1 = time.time()
        ret, frame = cap.read()
        if ret:
            print('[INFO] count frame: {}/{}'.format(count_frame, count))
            
            # detection privacy using SSD
            det_s = time.time()
            mask = detect_person(frame, datamean, model, args.postproc)
            det_e = time.time()
            print('[INFO] detection time per frame: {}'.format(det_e - det_s))

            # Inpainting using glcic
            if mask.max() > 0:
                # pre padding
                origin = frame.shape
                i, j, k = np.where(mask>=10)
                if i.max() > origin[0] - 5 and j.max() > origin[1] - 5:
                    print('[INFO] prepadding images...')
                    frame, mask = pre_padding(frame, mask, j, i, origin)

                # pre support
                large_thresh = 200
                rec = detect_large_mask(mask, large_thresh)
                n_input = frame.copy()
                n_mask = mask.copy()

                if rec != []:
                    print('[INFO] sparse patch...')
                    input256 = cv2.resize(n_input, (256, 256))
                    mask256 = cv2.resize(n_mask, (256, 256))
                    out256 = gl_inpaint(input256, mask256, datamean, model, args.postproc, device)
                    out256 = cv2.resize(out256, (origin[1], origin[0]))
                    out256 = (out256 * 255).astype('uint8')
                    n_input, n_mask = sparse_patch(n_input, out256, n_mask, rec, [256, 256])

                output = gl_inpaint(n_input, n_mask, datamean, model, args.postproc, device)
                output = output * 255 # innormalization
                output = output.astype('uint8')
            else:
                output = frame
            inp_e = time.time()
            print('[INFO] inpainting time per frame: {}'.format(inp_e - det_e))

            # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            concat = cv2.vconcat([frame, output])
            video.append(concat)
            cv2.imwrite('{}out_{}.png'.format(args.save_cap_dir, count_frame), output)
        else:
            break
        t2 = time.time()
        print('[INFO] elapsed time per frame: {}'.format(t2 - t1))
        count_frame += 1
        print('')
    t3 = time.time()
    print('[INFO] total elapsed time for ssd and inpaint: {}'.format(t3 - t0))
    cap.release()
    cv2.destroyAllWindows()

    video_name = args.video.split('/')[3].split('.')[0]
    outfile = '../data/video/out{}_{}.avi'.format(args.output, video_name)
    fps = args.fps
    codecs = 'H264'

    ret, frames, height, width, ch = np.array([video]).shape

    fourcc = cv2.VideoWriter_fourcc(*codecs)
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for i in range(frames):
        writer.write(video[i])
    writer.release()
    t4 = time.time()
    print('[INFO] elapsed time for saving video: {}'.format(t4 - t3))
