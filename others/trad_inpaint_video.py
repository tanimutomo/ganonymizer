import numpy as np
import argparse
import cv2
import time


def detect_person(image, datamean, model, postproc):
    # print('image.shape: ', image.shape)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (512, 512)), 1.0, (512, 512), 127.5)

    # print('[INFO] computing object detections...')
    net.setInput(blob)
    detections = net.forward()
# detections.shape = (1, 1, the number of the detected objects, 7)
# detections[0, 0, i, :] = [0, class, conf, upleft_x, upleft_y, bottomright_x, bottomright_y]
    mask = np.zeros((image.shape[0], image.shape[1], 1))
    # mask = np.zeros((image.shape[0], image.shape[1], 3))

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence(i.e., probability) associated with prediciton
        idx = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if idx == 2 or idx == 6 or idx == 7 or idx == 15 or idx == 14:
            # only person, car, bicycle, motorbike, bus
            # extract the index of the class label from the detections, then compute the (x, y)-coordinates of the bounding box for the object.
            # filter out weak detections by ensuring the confidence is greater than the minimum confidence
            if confidence > args.confidence:
                # print('detections.shape: ', detections.shape)
                # print('detections[3]: ', detections[0, 0, i, :])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # print('box.shape: ', box.shape)
                (start_x, start_y, end_x, end_y) = box.astype('int')
                label = '{}: {:2f}%'.format(CLASSES[idx], confidence * 100)
                print('[DETECT] {}'.format(label))
                
                # mosaic
                cut_img1 = image[start_y:end_y, start_x:end_x]
                # print('cut_img.shape: ', cut_img1.shape)
                if cut_img1.shape[0] > 1 and cut_img1.shape[1] > 1:
                    mask[start_y:end_y, start_x:end_x] = np.ones((cut_img1.shape[0], cut_img1.shape[1], 1)) * 255
                    # mask[start_y:end_y, start_x:end_x] = np.ones((cut_img1.shape[0], cut_img1.shape[1], 3)) * 255
                # show the output image

    mask = mask.astype('uint8')
    image = image.astype('uint8')
    # print('image.shape: {}'.format(image.shape))
    print('mask.shape: {}'.format(mask.shape))
    inpaint = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
    # if mask.max() > 0:
    #     inpaint = gl_inpaint(image, mask, datamean, model, postproc)
    # else:
    #     inpaint = image
    # cv2.imshow('Output', inpaint)
    return inpaint, mask

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='../data/video/')
parser.add_argument('--video_name', type=str, default='short_REC_170511_092456.avi')
parser.add_argument('--prototxt', default='./cfgs/deploy.prototxt', help='path to Caffe deploy prototxt file')
parser.add_argument('--model', default='./weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel', help='path to Caffe pre-trained file')
parser.add_argument('--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
parser.add_argument('--show', action='store_true')
parser.add_argument('--postproc', action='store_true')

# initialize the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


if __name__ == '__main__':
    args = parser.parse_args()
    file_dir = args.video_dir
    file_name = args.video_name
    video = []
    count_frame = 1
    print('[INFO] loading video...')
    cap = cv2.VideoCapture(file_dir + file_name)
    # load our serialized model from disk
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))

    t0 = time.time()

    # load Completion Network
    print('[INFO] loading model...')
    data = load_lua('./gl_gan/completionnet_places2.t7')
    model = data.model
    model.evaluate()
    datamean = data.mean

    while(cap.isOpened()):
        print('[INFO] count frame: {}/{}'.format(count_frame, count))
        t1 = time.time()
        ret, frame = cap.read()
        if ret:
            # frame = frame.astype('float32')
            # print('frame.type: {}'.format(frame.dtype))
            output, mask = detect_person(frame, datamean, model, args.postproc)
            # output = output.astype('float32')
            # print('output.type: {}'.format(output.dtype))
            # print('frame.shape: {}'.format(frame.shape))
            # print('frame: {}'.format(frame))
            # print('output: {}'.format(output))
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            concat = cv2.vconcat([frame, output])
            if args.show:
                cv2.imshow('Output', concat)
            video.append(concat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # cv2.imwrite('{}in_{}.png'.format(file_dir, count_frame), frame)
            # cv2.imwrite('{}m_{}.png'.format(file_dir, count_frame), mask)
            # cv2.imwrite('{}out_{}.png'.format(file_dir, count_frame), output)
        else:
            break
        t2 = time.time()
        print('[INFO] elapsed time per frame: {}'.format(t2 - t1))
        count_frame += 1
        print('')
    t3 = time.time()
    print('[INFO] total elapsed time for ssd and inpaint: {}'.format(t3 - t0))
    cap.release()
    cv2.destroyAllWindows()

    outfile = '{}out7_{}'.format(file_dir, file_name)
    fps = 20.0
    codecs = 'H264'

    # print(np.array([video]).shape)
    ret, frames, height, width, ch = np.array([video]).shape

    fourcc = cv2.VideoWriter_fourcc(*codecs)
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for i in range(frames):
        writer.write(video[i])
    writer.release()
    t4 = time.time()
    print('[INFO] elapsed time for saving video: {}'.format(t4 - t3))
