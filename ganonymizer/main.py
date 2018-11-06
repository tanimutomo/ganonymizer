import os

from .src.executer import Executer

def main():
    config = get_config()
    executer = Executer(config)
    executer.execute()


def get_config():
    config = {
            'video': '', # os.path.join(os.getcwd(), 'ganonymizer/data/videos/ex_small6_inter10_noon.avi'),
            # The input image, when you apply GANonymizer to an image.
            'image': os.path.join(os.getcwd(), 'ganonymizer/data/images/tmp1.png'),
            'output': '0',

            # minimum probability to filter weak detections
            'conf': 0.5,
            'nms': 0.4,
            # The threshold for PMD processing
            'large_thresh': 120,
            # The threshold for prepadding processing
            'prepad_thresh': 4,
            'fps': 7.0,
            'show': False,
            'postproc': False,

            # The mask image, when you apply Only reconstruction to an image.
            'mask': 'path_to_mask_image',
            # The [ulx,uly,rdx,rdy] of the mask you create
            'manual_mask': [],
            # The [position(edge/corner), distance(between edges), size] of the mask you want to create
            'edge_mask': [],
            # The size of the mask you create
            'center_mask': 0,

            # Write the bouding box at the reconstruction part
            'boxline': 3,
            'save_outframe': None,
            # {dir,filename(with extention)} that you want to save output image
            'save_outimage': None,
            'concat_inout': True,

            # path to Caffe deploy prototxt file
            'detect_cfgs': os.path.join(os.getcwd(), 'ganonymizer/src/detection/yolov3/cfgs/yolov3.cfg'),
            # path to Caffe pre-trained
            'detect_weights': os.path.join(os.getcwd(), 'ganonymizer/src/detection/yolov3/weights/yolov3.weights'),
            'inpaint_weights': os.path.join(os.getcwd(), 'ganonymizer/src/inpaint/glcic/weights/completionnet_places2.pth')
            }

    return config

if __name__ == '__main__':
    main()


# from inpaint.glcic.inpaint import gl_inpaint
# from inpaint.glcic.utils import *
# from inpaint.glcic.completionnet_places2 import completionnet_places2
# from detection.ssd.ssd512 import detect
# from utils.preprocessing import *
# from utils.create_mask import *
# from utils.design import *

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video', type=str)
#     parser.add_argument('--image', type=str, help='The input image, when you apply GANonymizer to an image.')
#     parser.add_argument('--output', default='')
# 
#     parser.add_argument('--conf', type=float, default=0.15, help='minimum probability to filter weak detections')
#     parser.add_argument('--large_thresh', type=int, default=120, help='The threshold for PMD processing')
#     parser.add_argument('--prepad_thresh', type=int, default=4, help='The threshold for prepadding processing')
#     parser.add_argument('--fps', type=float, default=10.0)
#     parser.add_argument('--show', action='store_true')
#     parser.add_argument('--postproc', action='store_true')
#     parser.add_argument('--cuda', default=None)
# 
#     parser.add_argument('--mask', type=str, help='The mask image, when you apply Only reconstruction to an image.')
#     parser.add_argument('--manual_mask', type=str, help='The ulx,uly,rdx,rdy of the mask you create')
#     parser.add_argument('--center_mask', type=int, help='The size of the mask you create')
#     parser.add_argument('--edge_mask', type=str, help='The position(edge/corner),distance(between edges),size of the mask you want to create')
#     parser.add_argument('--boxline', type=int, help='Write the bouding box at the reconstruction part')
#     parser.add_argument('--save_outframe', default=None)
#     parser.add_argument('--save_outimage', type=str, default=None, help='{dir,filename(with extention)} that you want to save output image')
# 
#     parser.add_argument('--prototxt', default='./detection/ssd/cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
#     parser.add_argument('--model', default='./detection/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained')
#     parser.add_argument('--inp_param', default='./inpaint/glcic/completionnet_places2.pth')
# 
#     args = parser.parse_args()
# 
#     return args


