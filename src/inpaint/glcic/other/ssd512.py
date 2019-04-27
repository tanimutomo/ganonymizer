import numpy as np
import cv2

def detect(image, net, conf, rec):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (512, 512)), 1.0, (512, 512), 127.5)

    # print('[INFO] computing object detections...')
    net.setInput(blob)
    detections = net.forward()
    mask = np.zeros((image.shape[0], image.shape[1], 3))

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence(i.e., probability) associated with prediciton
        idx = int(detections[0, 0, i, 1])
        confidence = detections[0, 0, i, 2]

        if idx == 2 or idx == 6 or idx == 7 or idx == 15 or idx == 14:
            # only person, car, bicycle, motorbike, bus
            if confidence > conf:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype('int')
                label = '{}: {:2f}%'.format(CLASSES[idx], confidence * 100)
                print('[DETECT] {}'.format(label))
                
                # mosaic
                rec.append([start_y, start_x, end_y - start_y, end_x - start_x])
                cut_img1 = image[start_y:end_y, start_x:end_x]
                if cut_img1.shape[0] > 1 and cut_img1.shape[1] > 1:
                    mask[start_y:end_y, start_x:end_x] = np.ones((cut_img1.shape[0], cut_img1.shape[1], 3)) * 255
                    mask = mask.astype('uint8')

    mask = mask.astype('uint8')
    # inpaint = cv2.inpaint(image, mask, 1, cv2.INPAINT_NS)
    return mask, rec

