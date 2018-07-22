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
parser.add_argument('--input', default='./ex_images/in_25.png', help='Input image')
parser.add_argument('--mask', default='./ex_images/mask4.png', help='Mask image')
parser.add_argument('--output', type=str, required=True, help='Output file name')
parser.add_argument('--cuda', type=str, default='0')
parser.add_argument('--model_path', default='completionnet_places2.t7', help='Trained model')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU')
parser.add_argument('--postproc', default=False, action='store_true',
                    help='Disable post-processing')
# print(args)

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
        # generate random holes
        M = torch.FloatTensor(1, I.size(1), I.size(2)).fill_(0)
        nHoles = np.random.randint(1, 4)
        print('[INFO] mask is not recognized...')
        print(nHoles)
        print('w: ', I.size(2))
        print('h: ', I.size(1))
        for _ in range(nHoles):
            mask_w = np.random.randint(32, 128)
            mask_h = np.random.randint(32, 128)
            assert I.size(1) > mask_h or I.size(2) > mask_w
            px = np.random.randint(0, I.size(2)-mask_w)
            py = np.random.randint(0, I.size(1)-mask_h)
            M[:, py:py+mask_h, px:px+mask_w] = 1


    for i in range(3):
        I[i, :, :] = I[i, :, :] - datamean[i]

# make mask_3ch
    M_3ch = torch.cat((M, M, M), 0)

    Im = I * (M_3ch*(-1)+1)

# set up input
    input = torch.cat((Im, M), 0)
    input = input.view(1, input.size(0), input.size(1), input.size(2)).float()

    if torch.cuda.is_available():
        print('using GPU...')
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

    print(out.shape)
    out = out.view(out.shape[1], out.shape[2], out.shape[3])
    out = np.array(out.cpu().detach()).transpose(1, 2, 0)
    out = out[:, :, [2, 1, 0]]
    print(out.shape)
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
    print(device)

    print('loading model...')
    # load Completion Network
    model = completionnet_places2
    param = torch.load('./completionnet_places2.pth')
    model.load_state_dict(param)
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)
    # data = load_lua('completionnet_places2.t7')
    # model = data.model
    # model.evaluate()

    print('loading images...')
    input_img = cv2.imread(args.input)
    mask_img = cv2.imread(args.mask)
    print(input_img.shape, mask_img.shape)
    print('processing images...')
    out = gl_inpaint(input_img, mask_img, datamean, model, args.postproc, device)
    # print('mask.shape: {}'.format(mask_img.shape))
    # print('mask.max: {}'.format(mask_img.max()))
    # print('mask.min(): {}'.format(mask_img.min()))

    # save images
    print('inpainting input image...')
    out_tensor = torch.from_numpy(cvimg2tensor(out))
    print('save images...')
    vutils.save_image(out_tensor, 'out.png', normalize=True)
    cv2.imwrite(args.output, out * 255)
    # vutils.save_image(Im, 'masked_input.png', normalize=True)
    # vutils.save_image(M_3ch, 'mask.png', normalize=True)
    # vutils.save_image(res, 'res.png', normalize=True)
    print('Done')
