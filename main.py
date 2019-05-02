import os

from src.executer import Executer
from src.utils.util import Config
from src.utils.crop_img import imcrop


def evaluate():
    from src.utils.calc_psnr_ssim import AverageMeter, PSNRSSIMCalcurator
    data_dir = 'data/videos/noon'
    calcurator = PSNRSSIMCalcurator(data_dir)
    calcurator.calcurate()

def v2f():
    from src.utils.video2frame import vid2frm
    infile = 'data/videos/noon/out_gfp_on_inter10_noon.avi'
    out_dir = 'data/videos/noon/out_gfp_on'
    vid2frm(infile, out_dir)

def main():
    config = Config(get_config())
    executer = Executer(config)
    executer.execute()

def create_exp_figs(crop=False):
    esp_px_config = Config(get_config())
    esp_px = ['edge', 'opposite', 'random', 'random_pick']
    for name in esp_px:
        esp_px_config.output = 'esp_px_{}'.format(name)
        esp_px_config.prepad_px = name
        esp_px_config.edge_mask = [0, 1, 120]
        esp_px_config.det = 'data/experiment_fig_with_line'
        executer = Executer(esp_px_config)
        executer.execute()
        if crop:
            imcrop(
                    os.path.join(os.getcwd(), esp_px_config.det, 
                        'out_{}_{}'.format(esp_px_config.output, 
                            esp_px_config.image.split('/')[-1])),
                    'left',
                    200
                    )

    gfp_div_config = Config(get_config())
    gfp_div = [[4, 9, 16], ['thin', 'normal', 'thick']]
    for num in gfp_div[0]:
        for wid in gfp_div[1]:
            gfp_div_config.output = 'gfp_div_{}_{}'.format(num, wid)
            gfp_div_config.pmd_div_num = num
            gfp_div_config.lattice_width = wid
            gfp_div_config.center_mask = 300
            gfp_div_config.det = 'data/experiment_fig_with_line'
            executer = Executer(gfp_div_config)
            executer.execute()
            if crop:
                imcrop(
                        os.path.join(os.getcwd(), gfp_div_config.det, 
                            'out_{}_{}'.format(gfp_div_config.output, 
                                gfp_div_config.image.split('/')[-1])),
                        'center',
                        400
                        )

    esp_thresh_config = Config(get_config())
    esp_thresh = [0, 1, 2, 3, 4, 5, 6, 10, 20]
    for dst in esp_thresh:
        esp_thresh_config.output = 'esp_thresh_{}'.format(dst)
        esp_thresh_config.edge_mask = [0, dst, 120]
        esp_thresh_config.prepad_thresh = -4
        esp_thresh_config.det = 'data/experiment_fig_with_line'
        executer = Executer(esp_thresh_config)
        executer.execute()
        if crop:
            imcrop(
                    os.path.join(os.getcwd(), esp_thresh_config.det, 
                        'out_{}_{}'.format(esp_thresh_config.output, 
                            esp_thresh_config.image.split('/')[-1])),
                    'left',
                    200
                    )

    gfp_thresh_config = Config(get_config())
    gfp_thresh = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 250, 300]
    for size in gfp_thresh:
        gfp_thresh_config.output = 'gfp_thresh_{}'.format(size)
        gfp_thresh_config.center_mask = size
        gfp_thresh_config.large_thresh = 120000
        gfp_thresh_config.det = 'data/experiment_fig_with_line'
        executer = Executer(gfp_thresh_config)
        executer.execute()
        if crop:
            imcrop(
                    os.path.join(os.getcwd(), gfp_thresh_config.det, 
                        'out_{}_{}'.format(gfp_thresh_config.output, 
                            gfp_thresh_config.image.split('/')[-1])),
                    'center',
                    300
                    )


def get_config():
    config = {
            # For Input and Output
            'video': '', # os.path.join(os.getcwd(), 'data/videos/half_inter10_noon.avi'),

            # The input image, when you apply to an image.
            'image': os.path.join(os.getcwd(), 'data/images/ex04.png'),
            'det': 'data/images',

            'output': 'boxline',

            # THe experiment for the PMD
            'pmd_div_num': 9, # choose in [4, 9, 16]
            'lattice_width': 'normal', # choose in ['thin', 'normal', 'thick']

            # The experiment for pre padding
            # choose in ['default', 'random', 'random_pick', 'edge', 'opposite']
            # default is 'edge'
            'prepad_px': 'default',

            # The int list [position(0:edge, 1:corner), distance(between edges), size] of the mask you want to create
            'edge_mask': [],

            # The size of the mask you create
            'center_mask': 300,

            # For network configuration
            'segmentation': False,
            # minimum probability to filter weak detections
            'conf': 0.5,

            'nms': 0.4,

            # The threshold for PMD processing
            'large_thresh': 120,

            # The threshold for prepadding processing
            'prepad_thresh': 4,

            'fps': 5.0,

            'show': False,

            'postproc': False,


            # For mask setting (this configs mainly for evaluating GANonymizer, ESP and GFP)

            # The mask image, when you apply Only reconstruction to an image.
            'mask': 'path_to_mask_image',

            # The [ulx,uly,rdx,rdy] of the mask you create
            'manual_mask': [],

            # select from [edge, large, None]
            # make a random mask and save its mask for evaluating the ESP or GFP
            'random_mask': None,

            # if you use local masks in video processing, specify the directory where series masks are saved.
            # the mask filename should be mask_{count_num}.png.
            # Note that this {count_num} start 1, not 0.
            # if you dont't use local masks, set None.
            'use_local_masks': None, # 'data/videos/noon/large_mask',


            # For design of the output image or video

            # Write the bouding box at the reconstruction part
            'boxline': 1,

            'concat_all': False,

            'concat_inout': True,

            'detection_summary_file': None, #'data/videos/noon/half_detection.txt',

            # For saving images and videos

            # {dir,filename(with extention)} that you want to save output image
            'save_outframe': None,

            # {dir,filename(with extention)} that you want to save image
            'save_outimage': None,

            # the dir where you want to save mask images. if you don't save the mask, set None.
            'save_mask': None, # 'data/videos/noon/large_mask',

            # For network pretrained models path

            # path to Caffe deploy prototxt file
            'detect_cfgs': os.path.join(os.getcwd(), 'src/detection/yolov3/cfgs/yolov3.cfg'),

            # path to Caffe pre-trained
            'detect_weights': os.path.join(os.getcwd(), 
                'src/detection/yolov3/weights/yolov3.weights'),

            'inpaint_weights': os.path.join(os.getcwd(), 
                'src/inpaint/glcic/weights/completionnet_places2.pth'),

            'segmentation_weights': os.path.join(os.getcwd(), 
                'src/segmentation/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth'),

            'resnet_type': 18,

            'resnet_path': os.path.join(os.getcwd(),
                'src/segmentation/deeplabv3/pretrained_models/resnet')
            }

    return config

if __name__ == '__main__':
    create_exp_figs(crop=True)
    # main()
    # v2f()
    # evaluate()


