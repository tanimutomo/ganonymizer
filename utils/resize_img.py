import cv2
import numpy as np
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=str)
    parser.add_argument('--factor', type=str)
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    return args

def resize(args):
    img = cv2.imread(args.file)
    if args.shape is not None:
        shape = list(map(int, args.shape.split(',')))
        print(shape)
        img = cv2.resize(img, shape)
    elif args.factor is not None:
        img = cv2.resize(img, None, fx=args.factor, fy=args.factor)
    
    file = args.file.split('/')
    name = file[-1]
    dir = '/'.join(file.remove(name)) + '/'
    cv2.imwrite(dir+'resize_'+name, img)

if __name__ == '__main__':
    args = get_parser()
    resize(args)


