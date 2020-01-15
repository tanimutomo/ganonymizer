import cv2
import torch

from .inpaint.glcic.completionnet_places2 import completionnet_places2
from .detection.yolov3.darknet import Darknet
from .segmentation.deeplabv3.model.deeplabv3 import DeepLabV3

def set_networks(config, device):
    print('[INFO] Loading model...')
    # detect_model = cv2.dnn.readNetFromCaffe(config.detect_cfgs, config.detect_weights)
    #Set up the neural network
    if config.segmentation:
        detecter = DeepLabV3(config.resnet_type, config.resnet_path, device)
        param = torch.load(config.segment_weights, map_location=device)
        detecter.load_state_dict(param)
        detecter.to(device) # (set in evaluation mode, this affects BatchNorm and dropout)
        detecter.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    else:
        detecter = Darknet(config.detect_cfgs)
        detecter.load_weights(config.detect_weights)
        detecter.to(device)
        detecter.eval()

    inpainter = completionnet_places2
    param = torch.load(config.inpaint_weights)
    inpainter.load_state_dict(param)
    inpainter.eval()
    inpainter.to(device)
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    return detecter, inpainter, datamean

def set_device(gpu_id):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device is {}'.format(device))
    return device
