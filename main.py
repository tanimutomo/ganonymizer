import argparse
import os
import glob

from src.executer import Executer
# from src.utils.util import Config
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
    # config = Config(get_config())
    config = get_options()
    executer = Executer(config)
    executer.execute()


def execute_to_dir():
    # config = Config(get_config())
    config = get_options()
    config.det = 'data/noon_det_res'
    config.boxline = 1
    imlist = glob.glob('data/videos/noon/input/*.png')
    for impath in imlist:
        config.image = os.path.join(os.getcwd(), impath)
        executer = Executer(config)
        executer.execute()


def create_exp_figs(crop=False):
    # esp_px_config = Config(get_config())
    esp_px_config = get_options()
    esp_px = ['edge', 'opposite', 'random', 'random_pick']
    for name in esp_px:
        esp_px_config.output = 'esp_px_{}'.format(name)
        esp_px_config.prepad_px = name
        esp_px_config.edge_mask = [0, 1, 120]
        esp_px_config.det = 'data/white_cropped'
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

    # gfp_div_config = Config(get_config())
    gfp_px_config = get_options()
    gfp_div = [[4, 9, 16], ['thin', 'normal', 'thick']]
    for num in gfp_div[0]:
        for wid in gfp_div[1]:
            gfp_div_config.output = 'gfp_div_{}_{}'.format(num, wid)
            gfp_div_config.pmd_div_num = num
            gfp_div_config.lattice_width = wid
            gfp_div_config.center_mask = 300
            gfp_div_config.det = 'data/white_cropped'
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

    # esp_thresh_config = Config(get_config())
    esp_thresh_config = get_options()
    esp_thresh = [0, 1, 2, 3, 4, 5, 6, 10, 20]
    for dst in esp_thresh:
        esp_thresh_config.output = 'esp_thresh_corner_enhanced_3_{}'.format(dst)
        esp_thresh_config.edge_mask = [1, dst, 120]
        esp_thresh_config.prepad_thresh = -4
        esp_thresh_config.det = 'data/white_cropped'
        esp_thresh_config.enhance = 3
        executer = Executer(esp_thresh_config)
        executer.execute()
        if crop:
            imcrop(
                    os.path.join(os.getcwd(), esp_thresh_config.det, 
                        'out_{}_{}'.format(esp_thresh_config.output, 
                            esp_thresh_config.image.split('/')[-1])),
                    'bottom_left',
                    200
                    )

    # gfp_thresh_config = Config(get_config())
    gfp_thresh_config = get_options()
    gfp_thresh = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 250, 300]
    for size in gfp_thresh:
        gfp_thresh_config.output = 'gfp_thresh_enhanced_{}'.format(size)
        gfp_thresh_config.center_mask = size
        gfp_thresh_config.large_thresh = 120000
        gfp_thresh_config.det = 'data/white_cropped'
        gfp_thresh_config.enhance = 4
        executer = Executer(esp_thresh_config)
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


def get_options():
    parser = argparse.ArgumentParser(description="Execution Settings")

    ##### MAIN SETTINGS #####
    # For Input and Output
    parser.add_argument('--exec', type=str, required=True,
                        choices=['realtime_image', 'realtime_video', 'image', 'video'],
                        help='specify the execution type')
    parser.add_argument('--data_root', type=str, default='/media/jetson/TOSHIBA EXT')

    # settings for each execution type
    # realtime
    parser.add_argument('--realtime_show', action='store_true')

    # an input video path
    parser.add_argument('--video', type=str, default='data/videos/half_inter10_noon.avi')

    # an input image path
    parser.add_argument('--image', type=str, default='data/images/white.png',
                        help='input image path')


    ##### OUTPUT SETTINGS #####
    parser.add_argument('--resize_factor', type=float)
    parser.add_argument('--det', type=str, default='data/images',
                        help='output dir for image')
    parser.add_argument('--output', type=str, default='0')

    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--show', action='store_true')

    parser.add_argument('--output_type', type=str, default='raw',
                        choices=['raw', 'concat_all', 'concat_inout'])
    parser.add_argument('--detection_summary_file', type=str)

    # For saving images and videos
    parser.add_argument('--save_outframe', type=str,
                        help='{dir,filename(with extention)} that you want to save output image')
    parser.add_argument('--save_outimage', type=str,
                        help='{dir,filename(with extention)} that you want to save image')
    parser.add_argument('--save_mask', type=str, # 'data/videos/noon/large_mask',
                        help='the dir where you want to save mask images.'
                             'if you dont save the mask, set None.')


    ##### GANONYMIZER SETTINGS #####
    # The experiment for the PMD
    parser.add_argument('--pmd_div_num', type=int, default=9, choices=[4, 9, 16])
    parser.add_argument('--lattice_width', type=str, default='normal',
                        choices=['thin', 'normal', 'thick'])
    parser.add_argument('--prepad_px', type=str, default='default',
                        choices=['default', 'random', 'random_pick', 'edge', 'opposite'],
                        help='which prepad do you use')
    parser.add_argument('--large_thresh', type=int, default=120,
                        help='a threshold for applying pmd')
    parser.add_argument('--prepad_thresh', type=int, default=4,
                        help='a threshold for prepadding')


    ##### NETWORK CONFIGURATIONS #####
    parser.add_argument('--segmentation', action='store_true',
                        help='use semantic segmentation (deeplabv3)')

    # detection settings
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--nms', type=float, default=0.4)

    # GLCIC
    parser.add_argument('--postproc', action='store_true')

    # Network path
    # YOLO
    parser.add_argument('--detect_cfgs', type=str, default='src/detection/yolov3/cfgs/yolov3.cfg')
    parser.add_argument('--detect_weights', type=str,
                        default='src/detection/yolov3/weights/yolov3.weights')
    # GLCIC
    parser.add_argument('--inpaint_weights', type=str,
                        default='src/inpaint/glcic/weights/completionnet_places2.pth')
    # DeepLabV3
    parser.add_argument('--segmentation_weights', type=str,
                        default='src/segmentation/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth')
    parser.add_argument('--resnet_type', type=int, default=18)
    parser.add_argument('--resnet_path', type=str,
                        default='src/segmentation/deeplabv3/pretrained_models/resnet')


    ##### EXPERIMENT SETTINGS #####
    # mask settings
    parser.add_argument('--edge_mask', type=int, nargs=3, default=[],
                        help='[position(0:edge, 1:corner), distance(between edges), size] of the mask you want to create')
    parser.add_argument('--center_mask', type=int, default=0,
                        help='a size of the mask you create')
    parser.add_argument('--manual_mask', type=int, nargs=4, default=[],
                        help='[ulx,uly,rdx,rdy] of the mask you create')
    parser.add_argument('--random_mask', type=str, choices=['edge', 'large'],
                        help='make a random mask and save its mask for evaluating the ESP or GFP')
    parser.add_argument('--mask', type=str,
                        help='the mask image path (Image Processing Only)')
    parser.add_argument('--use_local_masks', type=str, 
                        help='if you use local masks in video processing,'
                             'specify the directory path where series masks are saved.'
                             'the mask filename should be mask_{count_num}.png.'
                             'Note that this {count_num} start 1, not 0.'
                             'if you dont use local masks, set None.')

    # For design of the output image or video
    parser.add_argument('--enhance', type=int, default=0,
                        help='a factor multiplied to output image for enhancing the experiment result')
    parser.add_argument('--boxline', type=int, default=0,
                        help='Write the bouding box at the reconstruction part')


    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    main()
    # v2f()
    # evaluate()
    # create_exp_figs(crop=True)
    # execute_to_dir()


