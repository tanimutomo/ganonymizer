import numpy as np
import argparse
import cv2
import time


def detect_person(image):
    # print('image.shape: ', image.shape)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # print('[INFO] computing object detections...')
    net.setInput(blob)
    detections = net.forward()
# detections.shape = (1, 1, the number of the detected objects, 7)
# detections[0, 0, i, :] = [0, class, conf, upleft_x, upleft_y, bottomright_x, bottomright_y]
    musk = np.zeros((image.shape[0], image.shape[1], 1))

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence(i.e., probability) associated with prediciton
        idx = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > args.confidence:
            # extract the index of the class label from the detections, then compute the (x, y)-coordinates of the bounding box for the object.
            if idx == 2 or idx == 15 or idx == 14:
                # only person
                # print('detections.shape: ', detections.shape)
                # print('detections[3]: ', detections[0, 0, i, :])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # print('box.shape: ', box.shape)
                (start_x, start_y, end_x, end_y) = box.astype('int')
                label = '{}: {:2f}%'.format(CLASSES[idx], confidence * 100)
                print('[INFO] {}'.format(label))
                
                # mosaic
                cut_img1 = image[start_y:end_y, start_x:end_x]
                # print('cut_img.shape: ', cut_img1.shape)
                if cut_img1.shape[0] > 1 and cut_img1.shape[1] > 1:
                    musk[start_y:end_y, start_x:end_x] = np.ones((cut_img1.shape[0], cut_img1.shape[1], 1)) * 255
                    musk = musk.astype('uint8')

                # show the output image
    musk = musk.astype('uint8')
    image = image.astype('uint8')
    inpaint = cv2.inpaint(image, musk, 1, cv2.INPAINT_NS)
    # cv2.imshow('Output', inpaint)
    return inpaint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='./images/example_12.jpg', help='path to input image')
    parser.add_argument('--video_dir', type=str, default='./video/')
    parser.add_argument('--video_name', type=str, default='REC_170511_092456.avi')
    parser.add_argument('--prototxt', default='./MobileNetSSD_deploy.prototxt.txt', help='path to Caffe deploy prototxt file')
    parser.add_argument('--model', default='./MobileNetSSD_deploy.caffemodel', help='path to Caffe pre-trained file')
    parser.add_argument('--confidence', type=float, default=0.2, help='minimum probability to filter weak detections')
    args = parser.parse_args()

    # initialize the list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class.
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print('[INFO] loading model...')
    net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    file_dir = args.video_dir
    file_name = args.video_name
    video = []
    cap = cv2.VideoCapture(file_dir + file_name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            output = detect_person(frame)
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            concat = cv2.vconcat([frame, output])
            cv2.imshow('Output', concat)
            video.append(concat)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    outfile = '{}out2_{}'.format(file_dir, file_name)
    fps = 20.0
    codecs = 'H264'

    # print(np.array([video]).shape)
    ret, frames, height, width, ch = np.array([video]).shape

    fourcc = cv2.VideoWriter_fourcc(*codecs)
    writer = cv2.VideoWriter(outfile, fourcc, fps, (width, height))
    for i in range(frames):
        writer.write(video[i])
    writer.release()
