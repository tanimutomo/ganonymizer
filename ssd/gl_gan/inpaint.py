import argparse
import os
import torch
from torch.legacy import nn
from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils
from completionnet_places2 import completionnet_places2

# from gl_gan.utils import *
# from gl_gan.poissonblending import prepare_mask, blend

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./ex_images/in25.png', help='Input image')
parser.add_argument('--mask', default='./ex_images/mask4.png', help='Mask image')
parser.add_argument('--output', type=str, default='', help='Output file name')
parser.add_argument('--cuda', type=str, default='1')
parser.add_argument('--model_path', default='completionnet_places2.t7', help='Trained model')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU')
parser.add_argument('--postproc', default=False, action='store_true',
                    help='Disable post-processing')

def pre_padding(input, mask, vi, hi, origin):
    s_v, e_v, s_h, e_h = vi.min(), vi.max(), hi.min(), hi.max()
    v_l = [origin[1]-4, origin[1]-3, origin[1]-2, origin[1]-1, s_v-1, s_v-2, s_v-3, s_v-4]
    h_l = [origin[0]-4, origin[0]-3, origin[0]-2, origin[0]-1, s_h-1, s_h-2, s_h-3, s_h-4]
    if e_v > origin[1] - 5:
        pv1 = input[:, v_l[v_l.index(e_v)+1], :].reshape(-1, 1, 3)
        pv2 = input[:, v_l[v_l.index(e_v)+2], :].reshape(-1, 1, 3)
        pv3 = input[:, v_l[v_l.index(e_v)+3], :].reshape(-1, 1, 3)
        pv4 = input[:, v_l[v_l.index(e_v)+4], :].reshape(-1, 1, 3)
        pv = np.concatenate([pv1, pv2, pv3, pv4], axis=1)
        input = np.concatenate([input, pv], axis=1)
        mask = np.concatenate([mask, np.zeros(pv.shape, dtype='uint8')], axis=1)

    if e_h > origin[0] - 5:
        ph1 = input[h_l[h_l.index(e_h)+1], :, :].reshape(1, -1, 3)
        ph2 = input[h_l[h_l.index(e_h)+2], :, :].reshape(1, -1, 3)
        ph3 = input[h_l[h_l.index(e_h)+3], :, :].reshape(1, -1, 3)
        ph4 = input[h_l[h_l.index(e_h)+4], :, :].reshape(1, -1, 3)
        ph = np.concatenate([ph1, ph2, ph3, ph4], axis=0)
        input = np.concatenate([input, ph], axis=0)
        mask = np.concatenate([mask, np.zeros(ph.shape, dtype='uint8')], axis=0)

    return input, mask

# def interpolation(input, mask, 

def cut_padding(out, origin):
    if out.shape[0] != origin[0]:
        out = np.delete(out, [origin[0], origin[0]+1, origin[0]+2, origin[0]+3], axis=0)

    if out.shape[1] != origin[1]:
        out = np.delete(out, [origin[1], origin[1]+1, origin[1]+2, origin[1]+3], axis=1)

    return out

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

    if torch.cuda.is_available():
        print('[INFO] using GPU...')
        model.to(device)
        input = input.to(device)

# evaluate
    res = model.forward(input)

# make out
    for i in range(3):
        I[i, :, :] = I[i, :, :] + datamean[i]

    out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

# post-processing
    if postproc:
        print('[INFO] post-postprocessing...')
        target = input_img    # background
        source = tensor2cvimg(out.numpy())    # foreground
        mask = input_mask
        out = blend(target, source, mask, offset=(0, 0))
        # print(out)
    # print(out.shape)
    out = out[0]
    out = np.array(out.cpu().detach()).transpose(1, 2, 0)
    out = out[:, :, [2, 1, 0]]
    # print(out.shape)
    # out = out * 255
    # out = out.transpose((1, 2, 0)).astype(np.uint8)
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    # print('out.shape: {}'.format(out.shape))
    # print('out: {}'.format(out))

    return out


if __name__ == '__main__':
    from utils import *
    # from poissonblending import prepare_mask, blend
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')

    print('[INFO] loading model...')
    # load Completion Network
    model = completionnet_places2
    param = torch.load('./completionnet_places2.pth')
    model.load_state_dict(param)
    model.eval()
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    print('[INFO] loading images...')
    input_img = cv2.imread(args.input)
    mask_img = cv2.imread(args.mask)
    origin = input_img.shape
    i, j, k = np.where(mask_img>=10)
    if i.max() > origin[0] - 5 and j.max() > origin[1] - 5:
        print('[INFO] prepadding images...')
        input_img, mask_img = pre_padding(input_img, mask_img, j, i, origin)

    print('[INFO] processing images...')
    out = gl_inpaint(input_img, mask_img, datamean, model, args.postproc, device)

    if origin != input_img.shape:
        print('[INFO] cut padding images...')
        out = cut_padding(out, origin)

    # save images
    out_tensor = torch.from_numpy(cvimg2tensor(out))
    print('[INFO] save images...')
    in_file = args.input.split('/')[2].split('.')[0]
    m_file = args.mask.split('/')[2].split('.')[0]
    out_file = './ex_images/out{}_{}_{}.png'.format(args.output, in_file, m_file)
    cv2.imwrite(out_file, out * 255)
    print('[INFO] Done')
