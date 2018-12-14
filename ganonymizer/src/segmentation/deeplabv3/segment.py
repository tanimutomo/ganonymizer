# camera-ready

import os
import cv2
import copy
import torch
import pickle
import numpy as np
from scipy.ndimage.measurements import find_objects

from .model.deeplabv3 import DeepLabV3
from .utils.utils import label_img_to_color


def main():
    img = os.path.join(os.getcwd(), pj_name, img_dir, img_name)
    device, network = set()
    pred = detect_deeplabv3(img, device, network)
    mask = create_mask(pred)
    display(img, pred, mask)

def set():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network = DeepLabV3().to(device)
    network.load_state_dict(torch.load("my_deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth", map_location=device))
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

    return device, network


def detect_deeplabv3(input, network, device):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        # normalize the img (with mean and std for the pretrained ResNet):

        img = input/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (512, 1024, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 512, 1024))
        img = img.astype(np.float32)

        # convert numpy -> torch:
        img = torch.from_numpy(img) # (shape: (3, 512, 1024))

        img = torch.unsqueeze(img, 0) # (shape: (batch_size, 3, img_h, img_w))
        img = img.to(device) # (shape: (batch_size, 3, img_h, img_w))
        outputs = network(img) # (shape: (batch_size, num_classes, img_h, img_w))

        outputs = torch.squeeze(outputs, 0) # (shape: (num_classes, img_h, img_w))
        outputs = outputs.data.cpu().numpy() # (shape: (batch_size, num_classes, img_h, img_w))
        pred_label_img = np.argmax(outputs, axis=0) # (shape: (img_h, img_w))
        pred_label_img = pred_label_img.astype(np.uint8)

        # print(pred_label_img)
        # print(pred_label_img[100:200, 100:200])
        labels = np.reshape(pred_label_img, -1)
        # print(np.unique(labels))

        return pred_label_img


def calc_bbox(pred, mask, obj_rec):
    bbox_mask = np.where(mask[:,:,0] == 255, pred, 0)
    print(bbox_mask.shape)
    bbox = find_objects(bbox_mask)
    while True:
        try:
            bbox.remove(None)
        except:
            break
    print(bbox)

    for ys, xs in bbox:
        y = ys[0]
        x = xs[0]
        h = ys[1] - y
        w = xs[1] - x
        obj_rec.append([y, x, h, w])

    return obj_rec


def create_mask(pred):
    condition = (pred == 11) | (pred == 12) | (pred == 13) | (pred == 14) | \
            (pred == 15) | (pred == 16) | (pred == 17) | (pred == 18) | (pred == 19)
    mask = np.where(condition, 255, 0)
    mask = np.stack((mask, mask, mask), axis=2)
    mask = mask.astype('uint8')
    return mask


def display(img, pred_label_img, mask):
    img = cv2.imread(img, -1) # (shape: (512, 1024, 3))
    pred_label_img_color = label_img_to_color(pred_label_img)
    overlayed_img = 0.35*img + 0.65*pred_label_img_color
    overlayed_img = overlayed_img.astype(np.uint8)

    # TODO! do this using network.model_dir instead
    cv2.imwrite(os.path.join(os.getcwd(), pj_name, img_dir, 'pred_' + img_name), pred_label_img_color)
    cv2.imwrite(os.path.join(os.getcwd(), pj_name, img_dir, 'overlayed_' + img_name), overlayed_img)
    cv2.imwrite(os.path.join(os.getcwd(), pj_name, img_dir, 'mask_' + img_name), mask)


if __name__ == '__main__':
    pj_name = 'my_deeplabv3'
    img_dir = 'image'
    img_name = 'example_10.jpeg'
    main()
