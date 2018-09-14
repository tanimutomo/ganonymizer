import argparse
import os
import torch
from torch.legacy import nn
from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
# from torch.utils.serialization import load_lua
import torchvision.utils as vutils

from inpaint.glcic.completionnet_places2 import completionnet_places2
from inpaint.glcic.utils import *

from inpaint.glcic.poissonblending import prepare_mask, blend

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./ex_images/in25.png', help='Input image')
parser.add_argument('--mask', default=None, help='Mask image')
parser.add_argument('--conf', type=float, default=0.15)
parser.add_argument('--output', type=str, default='', help='Output file name')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--model_path', default='completionnet_places2.t7', help='Trained model')
parser.add_argument('--prototxt', default='../../detection/ssd/cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
parser.add_argument('--model', default='../../detection/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained file')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU')
parser.add_argument('--postproc', default=False, action='store_true',
                    help='Disable post-processing')

def gl_inpaint(input_img, mask, datamean, model, postproc, device):
# load data
    # print('image.shape: {}'.format(image.shape))
    # input_img = cv2.imread(image)
    I = torch.from_numpy(cvimg2tensor(input_img)).float().to(device)

    if mask.shape[2] == 3:
        input_mask = mask
        M = torch.from_numpy(
                    cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float().to(device)
        # print('M.shape: {}'.format(M.shape))
        M[M <= 0.2] = 0.0
        M[M > 0.2] = 1.0
        M = M.view(1, M.size(0), M.size(1))
        assert I.size(1) == M.size(1) and I.size(2) == M.size(2)
    
    else:
        print('[ERROR] Mask image is invalid')

    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]

# make mask_3ch
    M_3ch = torch.cat((M, M, M), 0)
    Im = I * (M_3ch*(-1)+1)

# set up input
    input = torch.cat((Im, M), 0)
    input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

    model.to(device)
    input = input.to(device)

# evaluate
    res = model.forward(input)

# make out
    for i in range(3):
        I[i, :, :] = I[i, :, :] + datamean[i]

    out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

    out = out[0]
    out = np.array(out.cpu().detach()).transpose(1, 2, 0)
    out = out[:, :, [2, 1, 0]]
    # cv2.imshow('out_before', out)
    # cv2.waitKey(0)

    # post-processing
    if postproc:
        print('[INFO] Post Processing...')
        target = input_img    # background
        mask = input_mask
        out = blend(target, out, mask, offset=(0, 0))
        out = out / 255
        # cv2.imshow('out_after', out)
        # cv2.waitKey(0)
#         print(out)
#         print(out.shape)

    return out


if __name__ == '__main__':
    from utils import *
    from pre_support import *
    from completionnet_places2 import completionnet_places2
    from other.ssd512 import detect
    from poissonblending import prepare_mask, blend

    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print('[INFO] device is {}'.format(device))

    print('[INFO] loading model...')
    # load Completion Network
    model = completionnet_places2
    param = torch.load('./completionnet_places2.pth')
    model.load_state_dict(param)
    model.eval()
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    print('[INFO] loading images...')
    input_img = cv2.imread(args.input)
    origin = input_img.shape
    if args.mask == None:
        obj_rec = []
        mask_img, obj_rec = detect(input_img, net, args.conf, obj_rec)
    else:
        mask_img = cv2.imread(args.mask)


    # Inpainting using glcic
    if mask_img.max() > 0:
        # pre padding
        origin = input_img.shape
        n_input = input_img.copy()
        n_mask = mask_img.copy()
        i, j, k = np.where(n_mask>=10)
        flag = {'hu':False, 'hd':False, 'vl':False, 'vr':False}

        if i.max() > origin[0] - 5 or j.max() > origin[1] - 5 or i.min() < 4 or j.min() < 4:
            print('[INFO] prepadding images...')
            n_input, n_mask, flag = pre_padding(n_input, n_mask, j, i, origin, flag)

        # pre support
        # rec = detect_large_mask(n_mask)

        if obj_rec != []:
            print('[INFO] pseudo mask division...')
            input256 = cv2.resize(n_input, (256, 256))
            mask256 = cv2.resize(n_mask, (256, 256))
            out256 = gl_inpaint(input256, mask256, datamean, model, args.postproc, device)
            out256 = cv2.resize(out256, (origin[1], origin[0]))
            out256 = (out256 * 255).astype('uint8')
            large_thresh = 200
            n_input, n_mask = sparse_patch(n_input, out256, n_mask, obj_rec, [256, 256], large_thresh)

        output = gl_inpaint(n_input, n_mask, datamean, model, args.postproc, device)

        # cut pre_padding
        if flag['hu'] or flag['hd'] or flag['vl'] or flag['vr']:
            output = cut_padding(output, origin, flag)

        output = output * 255 # innormalization
        output = output.astype('uint8')

    else:
        output = frame

    # n_input, n_mask = grid_interpolation(n_input, n_mask_img, rec)

    # resize to 256
    # if rec != []:
    #     print('[INFO] sparse patch...')
    #     # print(n_input[rec[0][0]:rec[0][0]+rec[0][2], rec[0][1]:rec[0][1]+rec[0][3], :])
    #     input256 = cv2.resize(n_input, (256, 256))
    #     mask256 = cv2.resize(n_mask, (256, 256))
    #     out256 = gl_inpaint(input256, mask256, datamean, model, args.postproc, device)
    #     cv2.imwrite('./ex_images/input256.png', input256 * 255)
    #     cv2.imwrite('./ex_images/out256.png', out256 * 255)
    #     cv2.imwrite('./ex_images/mask256.png', mask256 * 255)
    #     cv2.imshow('out256', out256)
    #     cv2.waitKey(0)
    #     out256 = cv2.resize(out256, (origin[1], origin[0]))
    #     out256 = (out256 * 255).astype('uint8')
    #     # print(out256[rec[0][0]:rec[0][0]+rec[0][2], rec[0][1]:rec[0][1]+rec[0][3], :])
    #     cv2.imshow('out256', out256)
    #     cv2.waitKey(0)
    #     n_input, n_mask = sparse_patch(n_input, out256, n_mask, rec, [256, 256])

    # cv2.imshow('out', out)
    # cv2.waitKey(0)

    # save images
    print('[INFO] save images...')
    in_file = args.input.split('/')[-1]
    # m_file = args.mask.split('/')[2].split('.')[0]
    # out256_file = './ex2_images/256_{}'.format(in_file)
    input_file = './ex2_images/{}'.format(in_file)
    out_file = './ex2_images/out{}_{}'.format(args.output, in_file)
    if args.mask == None:
        mask_file = './ex2_images/mask_{}'.format(in_file)
    # out_file = './ex_images/out{}_{}_{}.png'.format(args.output, in_file, m_file)
    # cv2.imwrite(out256_file, out256)
    cv2.imwrite(input_file, input_img)
    cv2.imwrite(mask_file, mask_img)
    cv2.imwrite(out_file, output)
    print('[INFO] Done')
