import cv2
import torch

from ..inpaint.glcic.completionnet_places2 import completionnet_places2
from ..detection.ssd.ssd512 import ssd_detect

def set_networks(detect_cfgs, detect_weights, inpaint_weights, device):
    print('[INFO] Loading model...')
    detect_model = cv2.dnn.readNetFromCaffe(detect_cfgs, detect_weights)

    model = completionnet_places2
    param = torch.load(inpaint_weights)
    model.load_state_dict(param)
    model.eval()
    model.to(device)
    datamean = torch.tensor([0.4560, 0.4472, 0.4155], device=device)

    return detect_model, model, datamean

def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Device is {}'.format(device))
    return device
