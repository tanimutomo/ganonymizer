import os 
import os.path as osp
import cv2 
import random 
import itertools
import numpy as np
import pandas as pd
import pickle as pkl

import torch 
import torch.nn as nn
from torch.autograd import Variable

from .utils.util import load_classes, write_results
from .darknet import Darknet
from .utils.preprocess import prep_image, inp_to_image


def yolo_detecter(img, model, conf, nms, rec, device):    
    yolov3_path = 'ganonymizer/src/detection/yolov3'
    images = img
    batch_size = 1
    confidence = conf
    nms_thesh = nms

    num_classes = 80
    classes = load_classes(os.path.join(
        os.getcwd(), yolov3_path, 'data/coco.names')) 

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #Detection phase
    imlist = []
    imlist.append(images)
        
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    im_dim_list = im_dim_list.to(device)
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1

    i = 0

    objs = {}
    
    for batch in im_batches:
        batch = batch.to(device)
        
        with torch.no_grad():
            prediction = model(Variable(batch), device)
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        
        if type(prediction) == int:
            i += 1
            return []

        prediction[:,0] += i*batch_size
          
        output = prediction
        print(output)
        write = 1
        
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        i += 1


    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
    colors = pkl.load(open(os.path.join(os.getcwd(), yolov3_path, "pallete"), "rb"))

    privacy = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck']

    for out in output:
        out = summary(out, classes)
        rec = selection(out, rec, privacy)

    return rec


def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


def summary(x, classes):
    left_up = x[1:3].int()
    right_down = x[3:5].int()
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    return [left_up, right_down, label]


def selection(x, rec, privacy):
    if x[2] in privacy:
        print('[DETECT] {}'.format(x[2]))
        up = x[0][1].item()
        left = x[0][0].item()
        height = (x[1][1] - up).item()
        width = (x[1][0] - left).item()
        rec.append([up, left, height, width])

    return rec



if __name__ == '__main__':
    from utils.util import load_classes, write_results
    from darknet import Darknet
    from utils.preprocess import prep_image, inp_to_image
    image = cv2.imread('imgs/dog.jpg')
    conf = 0.5
    nms = 0.4
    rec = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    detecter = Darknet('cfgs/yolov3.cfg')
    detecter.load_weights('weights/yolov3.weights')
    detecter.to(device)
    detecter.eval()

    rec = yolo_detecter(image, detecter, conf, nms, rec, device)
    print(rec)
