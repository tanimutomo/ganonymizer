import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='../images/example_12.jpg')
parser.add_argument('--prototxt', default='./cfgs/MobileNetSSD_deploy.prototxt.txt')
parser.add_argument('--model', default='./weights/MobileNetSSD_deploy.caffemodel')
parser.add_argument('--confidence', type=float, default=0.2)
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
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (512, 512)), 1.0, (512, 512), 127.5)
print(blob.shape)

print('[INFO] computing object detections...')
net.setInput(blob)
detections = net.forward()
print(detections.shape)

for i in np.arange(0, detections.shape[2]):
    # extract the confidence(i.e., probability) associated with prediciton
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is greater than the minimum confidence
    if confidence > args.confidence:
        # extract the index of the class label from the detections, then compute the (x, y)-coordinates of the bounding box for the object.
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_x, start_y, end_x, end_y) = box.astype('int')
        # display the prediction
        label = '{}: {:2f}%'.format(CLASSES[idx], confidence * 100)
        print('[INFO] {}'.format(label))
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), COLORS[idx], 2)
        y = start_y - 15 if start_y - 15 > 15 else start_y + 15
        cv2.putText(image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output image
        cv2.imshow('Output', image)
        cv2.waitKey(0)
