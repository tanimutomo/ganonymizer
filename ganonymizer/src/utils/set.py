import cv2
import torch

from ..inpaint.glcic.completionnet_places2 import completionnet_places2
from ..detection.yolov3.darknet import Darknet

def set_networks(detect_cfg, detect_weight, inpaint_weight, device):
    print('[INFO] Loading model...')
    # detect_model = cv2.dnn.readNetFromCaffe(detect_cfgs, detect_weights)
    #Set up the neural network
    detecter = Darknet(detect_cfg)
    detecter.load_weights(detect_weight)
    detecter.to(device)
    detecter.eval()

    inpainter = completionnet_places2
    param = torch.load(inpaint_weight)
    inpainter.load_state_dict(param)
    inpainter.eval()
    inpainter.to(device)
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    return detect_model, model, datamean

def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device is {}'.format(device))
    return device
