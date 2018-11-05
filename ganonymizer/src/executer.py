import os
import cv2
import time
import numpy as np

from .utils.set import set_networks, set_device
from .utils.utils import video_writer, load_video, adjust_imsize, concat_inout
from .utils.mask_design import create_mask, center_mask, edge_mask, create_boxline, write_boxline
from .utils.auxiliary_layer import detect_large_mask
from .processer import GANonymizer


class Executer:
    def __init__(self, config):
        self.video = config['video']
        self.image = config['image']
        self.output = config['output']
        self.detect_cfgs = config['detect_cfgs']
        self.detect_weights = config['detect_weights']
        self.inpaint_weights = config['inpaint_weights']

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
        self.concat_inout = config['concat_inout']

    def execute(self):
        device = set_device()
        detecter, inpainter, datamean = set_networks(
                self.detect_cfgs, self.detect_weights, self.inpaint_weights, device)
        self.ganonymizer = GANonymizer(self.conf, self.nms, self.postproc, self.large_thresh,
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
        input = image.copy()

        # process
        elapsed, output, image_designed = self.process_image(input, elapsed)

        if self.save_outimage is not None:
            dir = self.save_outimage.split(',')[0] + '/'
            name = self.save_outimage.split(',')[1] + '.png'
            cv2.imwrite(dir+name, output)
        elif self.concat_inout:
            concat = concat_inout(image, image_designed, output)
            in_name = self.image.split('/')[-1]
            save_path = os.path.join(os.getcwd(), 'ganonymizer/data/images/concat{}_{}'.format(self.output, in_name))
            cv2.imwrite(save_path, concat)
        else:
            # img_path = self.image.split('/')
            # dir = img_path.copy()
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
        print(self.video)
        cap, origin_fps, frames, width, height = load_video(self.video)
        print('after loading video')
        
        # video writer
        if self.concat_inout:
            writer = video_writer(self.video, self.output, self.fps, width*3, height*2)
        else:
            writer = video_writer(self.video, self.output, self.fps, width, height*2)

        print('after creating video writer')

        while(cap.isOpened()):
            print('')
            begin_process = time.time()
            ret, frame = cap.read()
            if ret:
                print('-----------------------------------------------------')
                print('[INFO] Count: {}/{}'.format(count, frames))

                # process
                input = frame.copy()
                elapsed, output, frame_designed = self.process_image(input, elapsed)

                # append frame to video
                if self.concat_inout:
                    concat = concat_inout(frame, frame_designed, output)
                else:
                    concat = np.concatenate([frame, output], axis=0)

                writer.write(concat)

                if self.save_outframe != None:
                    cv2.imwrite('{}out_{}.png'.format(self.save_outframe, count), output)

                # print the process info per iteration
                total_time, count = self.print_info_per_process(begin_process, elapsed, count, total_time, frames)

            else:
                break

        ### Stop video process
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


    def process_image(self, input, elapsed):
        obj_rec = []

        # detect
        if os.path.exists(self.mask):
            mask = cv2.imread(self.mask)
        elif len(self.manual_mask) > 0:
            mask, obj_rec = create_mask(input.shape, self.manual_mask)
        elif len(self.edge_mask) > 0:
            mask, obj_rec = edge_mask(input.shape, self.edge_mask)
        elif self.center_mask is not 0:
            mask, obj_rec = center_mask(input.shape, self.center_mask)
        else:
            obj_rec, elapsed[1] = self.ganonymizer.detect(input, obj_rec)
            mask = np.zeros((input.shape[0], input.shape[1], 3))
            mask = self.ganonymizer.create_detected_mask(input, mask, obj_rec)

        # print(obj_rec)
        # tmp = detect_large_mask(mask)
        cv2.imwrite(os.path.join(os.getcwd(), 'ganonymizer/data/images/mask.png'), mask)

        original = input.copy()
        origin_mask = mask.copy()
        if self.boxline > 0:
            boxline = np.zeros((input.shape))
            boxline = create_boxline(mask, obj_rec, boxline, self.boxline)

        # reconstruct
        output, elapsed[2], elapsed[3] = self.ganonymizer.reconstruct(input, mask, obj_rec)

        if self.boxline > 0:
            original = write_boxline(original, origin_mask, boxline)
            output = write_boxline(output, origin_mask, boxline)

        if self.show:
            disp = np.concatenate([original, output, origin_mask], axis=1)
            cv2.imshow('Display', disp)
            cv2.waitKey(0)

        return elapsed, output, original


    def print_info_per_process(self, begin, elapsed, count, total, frames):
        ### Print the elapsed time of processing
        elapsed[0] = time.time() - begin
        total[0] += elapsed[0]
        total[1] += elapsed[1]
        total[2] += elapsed[2]
        total[3] += elapsed[3]

        print('[TIME] Whole process time: {:.3f}'.format(elapsed[0]))
        print('-----------------------------------------------------')

        if count % 10 == 0 or count == frames:
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

