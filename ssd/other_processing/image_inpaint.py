import numpy as np
import argparse
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./images/example_12.jpg', help='path to input image')
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

# load the input image and construct an input blob for the image by resizing to a fixed 300x300 pixels and then normalizing it (note: normalization is done via the authors of the MobileNet SDD implementation)
image = cv2.imread(args.image)
print('image.shape: ', image.shape)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

print('[INFO] computing object detections...')
net.setInput(blob)
detections = net.forward()
# detections.shape = (1, 1, the number of the detected objects, 7)
# detections[0, 0, i, :] = [0, class, conf, upleft_x, upleft_y, bottomright_x, bottomright_y]
musk = np.zeros((image.shape[0], image.shape[1], 1))

for i in np.arange(0, detections.shape[2]):
    # extract the confidence(i.e., probability) associated with prediciton
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > args.confidence:
        # extract the index of the class label from the detections, then compute the (x, y)-coordinates of the bounding box for the object.
        if detections[0, 0, i, 1] == 15:
            # only person
            print('detections.shape: ', detections.shape)
            print('detections[3]: ', detections[0, 0, i, :])
            # idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            print('box.shape: ', box.shape)
            (start_x, start_y, end_x, end_y) = box.astype('int')
            # display the prediction
            label = '{}: {:2f}%'.format(CLASSES[15], confidence * 100)
            print('[INFO] {}'.format(label))
            # cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
            
            # mosaic
            cut_img = image[start_y:end_y, start_x:end_x]
            # print('cut_img.shape', cut_img.shape)
            # print('musk[start_y:end_y, start_x:end_x].shape', musk[start_y:end_y, start_x:end_x].shape)
            # print('np.ones(cut_img.shape)', np.ones((cut_img.shape[0], cut_img.shape[1], 1)))
            musk[start_y:end_y, start_x:end_x] = np.ones((cut_img.shape[0], cut_img.shape[1], 1)) * 255
            musk = musk.astype('uint8')
            # print('musk: ', musk)
            # print('image: ', image)
            # print('musk', musk)
            # print('cut_img.shape: ', cut_img.shape)
            # cut_img = cv2.resize(cut_img,((end_x + start_x)//100, (end_y + start_y)//100))
            # print('cut_img.shape: ', cut_img.shape)
            # cut_img = cv2.resize(cut_img,(end_x - start_x, end_y - start_y),cv2.INTER_NEAREST)
            # print('cut_img.shape: ', cut_img.shape)
            # print('image.shape: ', image.shape)
            # print('image[start_y:start_y+end_y,start_x:start_x+end_x].shape: ', image[start_y:end_y, start_x:end_x].shape)
            # image[start_y:end_y, start_x:end_x]=cut_img

            # y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            # cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # show the output image
inpaint = cv2.inpaint(image, musk, 10, cv2.INPAINT_NS)
cv2.imshow('Musk', musk)
cv2.imshow('Intput', image)
cv2.imshow('Output', inpaint)
cv2.waitKey(0)
