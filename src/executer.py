import os
import cv2
import time
import copy
import numpy as np
import datetime
import socket

from .set import set_networks, set_device
from .utils.util import video_writer, load_video, adjust_imsize, concat_all, \
        extend_rec, CreateRandMask, check_mask_position, find_bbox, enhance_img
from .utils.mask_design import create_mask, center_mask, edge_mask, \
        create_boxline, write_boxline, draw_rectangle
from .utils.auxiliary_layer import max_mask_size
from .processer import GANonymizer


class Executer:
    def __init__(self, config):
        self.config = config

    def execute(self):
        device = set_device()
        detecter, inpainter, datamean = set_networks(self.config, device)
        self.ganonymizer = GANonymizer(self.config, device, detecter, inpainter, datamean)

        if self.config.exec == 'realtime_image':
            self.realtime_image()
        elif self.config.exec == 'realtime_video':
            self.realtime_video()
        elif self.config.exec == 'video' and os.path.exists(self.config.video):
            self.video()
        elif self.config.exec == 'image' and os.path.exists(self.config.image):
            self.image()
        else:
            raise RuntimeError('[ERROR] Not selected an input source.')


    def realtime_image(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 50007))
            s.listen(1)

            while True:
                cap = cv2.VideoCapture(1)
                width = int(cap.get(3))
                height = int(cap.get(4))
                origin_fps = cap.get(5)

                if not cap.isOpened():
                    break

                count = 1
                elapsed = [0, 0, 0, 0]

                print('')
                conn, addr = s.accept()
                with conn:
                    while True:
                        data = conn.recv(1024)
                        if data:
                            break

                ret, origin_frame = cap.read()
                if not ret:
                    break

                # saved image name and path
                recv_data = data.decode(encoding='utf-8')
                # now = datetime.datetime.now()
                # image_name = '{0:%m%d_%H%M_%S}.png'.format(now)
                image_name = recv_data + '.png'
                save_path = os.path.join(self.config.data_root,
                                         'image',
                                         image_name)

                print('-----------------------------------------------------')
                if self.config.resize_factor:
                    rf = self.config.resize_factor
                    origin_frame = cv2.resize(origin_frame, dsize=None, fx=rf, fy=rf)

                # process
                frame = copy.deepcopy(origin_frame)
                elapsed, output, frame_designed = self.process_input(frame, elapsed, count=count)

                if elapsed[2] == 0:
                    output = origin_frame
                if self.config.output_type == 'concat_all':
                    output = concat_all(origin_frame, frame_designed, output)
                elif self.config.output_type == 'concat_inout':
                    output = np.concatenate([origin_frame, output], axis=0)

                cv2.imwrite(save_path, output)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cs:
                    cs.connect(('127.0.0.1', 50008))
                    cs.send(save_path.encode())

                cap.release()

    def realtime_video(self):
        cap = cv2.VideoCapture(1)
        width = int(cap.get(3))
        height = int(cap.get(4))
        origin_fps = cap.get(5)

        # create video writer
        now = datetime.datetime.now()
        video_name = '{0:%m%d_%H%M_%S}.avi'.format(now)
        writer = video_writer(video_name, False, self.config.fps, width, height)

        count = 1
        elapsed = [0, 0, 0, 0]
        total_time = [0, 0, 0, 0]

        while(cap.isOpened()):
            print('')
            begin_process = time.time()
            ret, origin_frame = cap.read()
            if ret:
                print('-----------------------------------------------------')

                # process
                frame = copy.deepcopy(origin_frame)
                elapsed, output, frame_designed = self.process_input(frame, elapsed, count=count)

                # append frame to video
                if self.config.output_type == 'concat_all':
                    output = concat_all(origin_frame, frame_designed, output)
                elif self.config.output_type == 'concat_inout':
                    output = np.concatenate([origin_frame, output], axis=0)

                writer.write(output)
                if self.config.realtime_show:
                    cv2.imshow('Ouput', output)
                    k = cv2.waitKey(1)
                    if k == 27:
                        break

                # print the process info per iteration
                total_time, count = self.print_info_per_process(begin_process,
                                                                elapsed,
                                                                count,
                                                                total_time)

            else:
                break

        ### Stop video process
        cap.release()
        writer.release()
        cv2.destroyAllWindows()


    def image(self):
        # whole, yolov3, glcic, reconst
        elapsed = [0, 0, 0, 0]
        image = cv2.imread(self.config.image)
        image = adjust_imsize(image)
        input = copy.deepcopy(image)

        # process
        if os.path.exists(self.config.mask):
            mask = cv2.imread(self.config.mask)
            elapsed, output, image_designed = self.process_input(input, elapsed, mask)
        else:
            elapsed, output, image_designed = self.process_input(input, elapsed)


        if self.config.save_outimage is not None:
            dir = self.config.save_outimage.split(',')[0] + '/'
            name = self.config.save_outimage.split(',')[1] + '.png'
            cv2.imwrite(dir+name, output)
        elif self.config.output_type == 'concat_all':
            concat = concat_all(image, image_designed, output)
            in_name = self.config.image.split('/')[-1]
            save_path = os.path.join(os.getcwd(), self.config.det, 
                    'concat_{}_{}'.format(self.config.output, in_name))
            cv2.imwrite(save_path, concat)
        else:
            in_name = self.config.image.split('/')[-1]
            save_path = os.path.join(os.getcwd(), self.config.det,
                    'out_{}_{}'.format(self.config.output, in_name))
            cv2.imwrite(save_path, output)


    def video(self):
        # set variables
        video = np.array([])
        count = 1

        # whole, yolov3, glcic, reconst
        elapsed = [0, 0, 0, 0]
        total_time = [0, 0, 0, 0]

        # video data
        cap, origin_fps, frames, width, height = load_video(self.config.video)
        
        # video writer
        if self.config.output_type == 'concat_all':
            writer = video_writer(self.config.video, self.config.output, self.config.fps, width*3, height*2)
        elif self.config.output_type == 'concat_inout':
            writer = video_writer(self.config.video, self.config.output, self.config.fps, width, height*2)
        else:
            writer = video_writer(self.config.video, self.config.output, self.config.fps, width, height)


        while(cap.isOpened()):
            print('')
            begin_process = time.time()
            ret, origin_frame = cap.read()
            if ret:
                print('-----------------------------------------------------')
                print('[INFO] Count: {}/{}'.format(count, frames))

                # process
                frame = copy.deepcopy(frame)
                if self.config.use_local_masks is not None:
                    mask = cv2.imread(os.path.join(self.config.use_local_masks, 'mask_{}.png'.format(count)))
                    elapsed, output, frame_designed, mask = self.process_input(frame, elapsed, mask=mask, count=count)
                else:
                    elapsed, output, frame_designed, mask = self.process_input(frame, elapsed, count=count)

                # append frame to video
                if self.config.output_type == 'concat_all':
                    output = concat_all(origin_frame, frame_designed, output)
                elif self.config.output_type == 'concat_inout':
                    output = np.concatenate([origin_frame, output], axis=0)

                writer.write(output)

                if self.config.save_outframe != None:
                    cv2.imwrite('{}out_{}.png'.format(self.config.save_outframe, count), output)
                if self.config.save_mask is not None:
                    cv2.imwrite(os.path.join(self.config.save_mask, 'mask_{}.png'.format(count)), mask)

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
        elif self.config.manual_mask:
            mask, obj_rec = create_mask(input.shape, self.config.manual_mask)
        elif self.config.edge_mask:
            mask, obj_rec = edge_mask(input.shape, self.config.edge_mask)
        elif self.config.center_mask != 0:
            mask, obj_rec = center_mask(input.shape, self.config.center_mask)
        elif self.config.segmentation:
            mask, obj_rec, elapsed[1] = self.ganonymizer.segment(input, obj_rec)
            # reconstruct
            # output, elapsed[2], elapsed[3] = self.ganonymizer.reconstruct(
            #         input, mask, obj_rec)
            # if self.config.boxline > 0:
            #     origin_mask = copy.deepcopy(mask)
            #     boxline = create_boxline(mask, obj_rec, self.config.boxline, original)
            #     original = write_boxline(original, origin_mask, boxline)

        else:
            obj_rec, elapsed[1], detected_obj = self.ganonymizer.detect(input, obj_rec, detected_obj)
            if not obj_rec:
                # No object is detected
                return elapsed, False, original

            obj_rec = extend_rec(obj_rec, input)
            mask = np.zeros((input.shape[0], input.shape[1], 3))
            mask = self.ganonymizer.create_detected_mask(input, mask, obj_rec)

            if self.config.random_mask in ['edge', 'large']:
                print('[INFO] Create random {} mask...'.format(self.config.random_mask))
                start_calc_random_mask = time.time()
                loop_count = 0
                rand_mask_creater = CreateRandMask(mask.shape[0], mask.shape[1])
                while True:
                    loop_count += 1
                    if self.config.random_mask == 'edge':
                        rand_mask_creater.edge_sampling()
                    elif self.config.random_mask == 'large':
                        rand_mask_creater.large_sampling()
                    obj_rec = []
                    rand_mask = np.zeros((input.shape[0], input.shape[1], 3))
                    rand_mask, obj_rec = rand_mask_creater.create_mask(rand_mask, obj_rec)

                    if check_mask_position(rand_mask, mask):
                        break
                    
                    if loop_count >= 100:
                        rand_mask = np.zeros((input.shape[0], input.shape[1], 3))
                        break

                if self.config.random_mask == 'edge':
                    print('[INFO] Random {} Mask: position: {} masksize: {} distance: {} loop: {}'.format(
                        self.config.random_mask, 
                        rand_mask_creater.position,
                        rand_mask_creater.masksize,
                        rand_mask_creater.distance,
                        loop_count
                        ))

                elif self.config.random_mask == 'large':
                    print('[INFO] Random {} Mask: position: {} masksize: {} loop: {}'.format(
                        self.config.random_mask, 
                        rand_mask_creater.position,
                        rand_mask_creater.masksize,
                        loop_count
                        ))


                mask = rand_mask
                # print('[TIME] CreateRandMask elapsed time: {:.3f}'.format(
                #     time.time() - start_calc_random_mask))

        origin_mask = copy.deepcopy(mask)

        if obj_rec != []:
            try:
                width_max, height_max = max_mask_size(mask)
            except:
                width_max, height_max = 0, 0
        else:
            width_max, height_max = 0, 0
        # cv2.imwrite(os.path.join(os.getcwd(), 'ganonymizer/data/images/mask.png'), mask)

        # origin_mask = copy.deepcopy(mask)
        # if self.config.boxline > 0:
        #     boxline = np.zeros((original.shape))
        #     boxline = create_boxline(mask, obj_rec, boxline, self.config.boxline)

        # reconstruct
        output, elapsed[2], elapsed[3] = self.ganonymizer.reconstruct(
                input, mask, obj_rec, width_max, height_max)

        if self.config.enhance > 0:
            output = enhance_img(output, self.config.enhance)

        if self.config.boxline > 0:
            # boxline = create_boxline(mask, obj_rec, self.config.boxline, original)
            # original = write_boxline(original, origin_mask, boxline)
            # output = write_boxline(output, origin_mask, boxline)
            output = draw_rectangle(output, obj_rec, self.config.boxline)

        if self.config.show:
            disp = np.concatenate([original, output, origin_mask], axis=1)
            cv2.imshow('Display', disp)
            cv2.waitKey(0)

        if self.config.detection_summary_file is not None:
            with open(self.config.detection_summary_file, mode='a') as f:
                f.write('\n\ncount: {}\n'.format(count))
                f.write('\n'.join(detected_obj))

        # return elapsed, output, original, origin_mask
        return elapsed, output, original


    def print_info_per_process(self, begin, elapsed, count, total, frames=-1):
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

