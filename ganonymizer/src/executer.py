import os
import cv2
import time
import copy
import numpy as np

from .utils.set import set_networks, set_device
from .utils.utils import video_writer, load_video, adjust_imsize, concat_all, extend_rec, CreateRandMask, check_mask_position, find_bbox
from .utils.mask_design import create_mask, center_mask, edge_mask, create_boxline, write_boxline
from .utils.auxiliary_layer import max_mask_size, detect_large_mask
from .processer import GANonymizer


class Executer:
    def __init__(self, config):
        self.video = config['video']
        self.image = config['image']
        self.output = config['output']
        self.segmentation = config['segmentation']
        self.detect_cfg = config['detect_cfgs']
        self.detect_weight = config['detect_weights']
        self.segmentation_weight = config['segmentation_weight']
        self.res_type = config['resnet_type']
        self.res_path = config['resnet_path']
        self.inpaint_weight = config['inpaint_weights']

        self.fps = config['fps']
        self.conf = config['conf']
        self.nms = config['nms']
        self.postproc = config['postproc']
        self.large_thresh = config['large_thresh']
        self.prepad_thresh = config['prepad_thresh']

        self.show = config['show']
        self.mask = config['mask']
        self.manual_mask = config['manual_mask']
        self.edge_mask = config['edge_mask']
        self.center_mask = config['center_mask']
        self.boxline = config['boxline']
        self.save_outframe = config['save_outframe']
        self.save_outimage = config['save_outimage']
        self.save_mask = config['save_mask']
        self.concat_all = config['concat_all']
        self.concat_inout = config['concat_inout']
        self.random_mask = config['random_mask']
        self.use_local_masks = config['use_local_masks']
        self.detection_summary_file = config['detection_summary_file']


    def execute(self):
        device = set_device()
        detecter, inpainter, datamean = set_networks(self.segmentation, self.detect_cfg, self.detect_weight, 
                self.segmentation_weight, self.res_type, self.res_path, self.inpaint_weight, device)
        self.ganonymizer = GANonymizer(self.segmentation, self.conf, self.nms, self.postproc, self.large_thresh,
                self.prepad_thresh, device, detecter, inpainter, datamean)

        if os.path.exists(self.video):
            self.apply_to_video()
        elif os.path.exists(self.image):
            self.apply_to_image()
        else:
            print('[ERROR] Not selected an input source.')


    def apply_to_image(self):
        # whole, yolov3, glcic, reconst
        elapsed = [0, 0, 0, 0]
        image = cv2.imread(self.image)
        image = adjust_imsize(image)
        input = copy.deepcopy(image)

        # process
        if os.path.exists(self.mask):
            mask = cv2.imread(self.mask)
            elapsed, output, image_designed = self.process_input(input, elapsed, mask)
        else:
            elapsed, output, image_designed = self.process_input(input, elapsed)


        if self.save_outimage is not None:
            dir = self.save_outimage.split(',')[0] + '/'
            name = self.save_outimage.split(',')[1] + '.png'
            cv2.imwrite(dir+name, output)
        elif self.concat_all:
            concat = concat_all(image, image_designed, output)
            in_name = self.image.split('/')[-1]
            save_path = os.path.join(os.getcwd(), 'ganonymizer/data/images/concat{}_{}'.format(self.output, in_name))
            cv2.imwrite(save_path, concat)
        else:
            # img_path = self.image.split('/')
            # dir = copy.deepcopy(img_path)
            # dir.pop()
            # dir = '/'.join(dir) + '/'
            # name = img_path[-1].split('.')[0]
            # ext = img_path[-1].split('.')[-1]
            # save_path = dir +  name + ext
            # print(save_path)
            in_name = self.image.split('/')[-1]
            save_path = os.path.join(os.getcwd(), 'ganonymizer/data/images/out{}_{}'.format(self.output, in_name))
            cv2.imwrite(save_path, output)


    def apply_to_video(self):
        # set variables
        video = np.array([])
        count = 1

        # whole, yolov3, glcic, reconst
        elapsed = [0, 0, 0, 0]
        total_time = [0, 0, 0, 0]

        # video data
        cap, origin_fps, frames, width, height = load_video(self.video)
        
        # video writer
        if self.concat_all:
            writer = video_writer(self.video, self.output, self.fps, width*3, height*2)
        elif self.concat_inout:
            writer = video_writer(self.video, self.output, self.fps, width, height*2)
        else:
            writer = video_writer(self.video, self.output, self.fps, width, height)


        while(cap.isOpened()):
            print('')
            begin_process = time.time()
            ret, frame = cap.read()
            if ret:
                print('-----------------------------------------------------')
                print('[INFO] Count: {}/{}'.format(count, frames))

                # process
                input = copy.deepcopy(frame)
                if self.use_local_masks is not None:
                    mask = cv2.imread(os.path.join(self.use_local_masks, 'mask_{}.png'.format(count)))
                    elapsed, output, frame_designed, mask = self.process_input(input, elapsed, mask=mask, count=count)
                else:
                    elapsed, output, frame_designed, mask = self.process_input(input, elapsed, count=count)

                # append frame to video
                if self.concat_all:
                    concat = concat_all(frame, frame_designed, output)
                elif self.concat_inout:
                    concat = np.concatenate([frame, output], axis=0)
                else:
                    concat = output

                writer.write(concat)

                if self.save_outframe != None:
                    cv2.imwrite('{}out_{}.png'.format(self.save_outframe, count), output)
                if self.save_mask is not None:
                    cv2.imwrite(os.path.join(self.save_mask, 'mask_{}.png'.format(count)), mask)

                # print the process info per iteration
                total_time, count = self.print_info_per_process(begin_process, elapsed, count, total_time, frames)

            else:
                break

        ### Stop video process
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


    def process_input(self, input, elapsed, mask=None, count=None):
        original = copy.deepcopy(input)
        obj_rec = []
        detected_obj = []

        # detect
        if mask is not None:
            print('[INFO] Use local masks')
            obj_rec = find_bbox(mask, obj_rec)
        elif len(self.manual_mask) > 0:
            mask, obj_rec = create_mask(input.shape, self.manual_mask)
        elif len(self.edge_mask) > 0:
            mask, obj_rec = edge_mask(input.shape, self.edge_mask)
        elif self.center_mask is not 0:
            mask, obj_rec = center_mask(input.shape, self.center_mask)
        elif self.segmentation:
            mask, obj_rec, elapsed[1] = self.ganonymizer.segment(input, obj_rec)
            # reconstruct
            # output, elapsed[2], elapsed[3] = self.ganonymizer.reconstruct(
            #         input, mask, obj_rec)
            # if self.boxline > 0:
            #     origin_mask = copy.deepcopy(mask)
            #     boxline = create_boxline(mask, obj_rec, self.boxline, original)
            #     original = write_boxline(original, origin_mask, boxline)

        else:
            obj_rec, elapsed[1], detected_obj = self.ganonymizer.detect(input, obj_rec, detected_obj)
            obj_rec = extend_rec(obj_rec, input)
            mask = np.zeros((input.shape[0], input.shape[1], 3))
            mask = self.ganonymizer.create_detected_mask(input, mask, obj_rec)

            if self.random_mask in ['edge', 'large']:
                print('[INFO] Create random {} mask...'.format(self.random_mask))
                start_calc_random_mask = time.time()
                loop_count = 0
                rand_mask_creater = CreateRandMask(mask.shape[0], mask.shape[1])
                while True:
                    loop_count += 1
                    if self.random_mask == 'edge':
                        rand_mask_creater.edge_sampling()
                    elif self.random_mask == 'large':
                        rand_mask_creater.large_sampling()
                    obj_rec = []
                    rand_mask = np.zeros((input.shape[0], input.shape[1], 3))
                    rand_mask, obj_rec = rand_mask_creater.create_mask(rand_mask, obj_rec)

                    if check_mask_position(rand_mask, mask):
                        break
                    
                    if loop_count >= 100:
                        rand_mask = np.zeros((input.shape[0], input.shape[1], 3))
                        break

                if self.random_mask == 'edge':
                    print('[INFO] Random {} Mask: position: {} masksize: {} distance: {} loop: {}'.format(
                        self.random_mask, 
                        rand_mask_creater.position,
                        rand_mask_creater.masksize,
                        rand_mask_creater.distance,
                        loop_count
                        ))

                elif self.random_mask == 'large':
                    print('[INFO] Random {} Mask: position: {} masksize: {} loop: {}'.format(
                        self.random_mask, 
                        rand_mask_creater.position,
                        rand_mask_creater.masksize,
                        loop_count
                        ))


                mask = rand_mask
                # print('[TIME] CreateRandMask elapsed time: {:.3f}'.format(
                #     time.time() - start_calc_random_mask))

        origin_mask = copy.deepcopy(mask)

        if obj_rec != []:
            width_max, height_max = max_mask_size(mask)
        else:
            width_max, height_max = 0, 0
        # cv2.imwrite(os.path.join(os.getcwd(), 'ganonymizer/data/images/mask.png'), mask)

        # origin_mask = copy.deepcopy(mask)
        # if self.boxline > 0:
        #     boxline = np.zeros((original.shape))
        #     boxline = create_boxline(mask, obj_rec, boxline, self.boxline)

        # reconstruct
        output, elapsed[2], elapsed[3] = self.ganonymizer.reconstruct(
                input, mask, obj_rec, width_max, height_max)

        if self.boxline > 0:
            boxline = create_boxline(mask, obj_rec, self.boxline, original)
            original = write_boxline(original, origin_mask, boxline)
            # output = write_boxline(output, origin_mask, boxline)

        if self.show:
            disp = np.concatenate([original, output, origin_mask], axis=1)
            cv2.imshow('Display', disp)
            cv2.waitKey(0)

        if self.detection_summary_file is not None:
            with open(self.detection_summary_file, mode='a') as f:
                f.write('\ncount: {}'.format(count))
                f.write('\n'.join(detected_obj))

        return elapsed, output, original, origin_mask


    def print_info_per_process(self, begin, elapsed, count, total, frames):
        ### Print the elapsed time of processing
        elapsed[0] = time.time() - begin
        total[0] += elapsed[0]
        total[1] += elapsed[1]
        total[2] += elapsed[2]
        total[3] += elapsed[3]

        print('[TIME] Whole process time: {:.3f}'.format(elapsed[0]))
        print('-----------------------------------------------------')

        if count % 100 == 0 or count == frames:
            print('')
            print('-----------------------------------------------------')
            print('[INFO] Time Summary')
            print('[TIME] YOLO-V3 average time per frame: {:.3f}'.format(total[1] / count))
            print('[TIME] GLCIC average time per frame: {:.3f}'.format(total[2] / count))
            print('[TIME] Reconstruction average time per frame: {:.3f}'.format(total[3] / count))
            print('[TIME] Whole process average time per frame: {:.3f}'.format(total[0] / count))
            print('-----------------------------------------------------')

        count += 1

        return total, count

