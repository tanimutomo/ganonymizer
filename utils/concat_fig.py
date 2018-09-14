import cv2
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=str)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    return args



def concat(args):
    # assert len(shape) < 3
    # sum = 0
    # for i in range(len(shape)):
    #     sum += shape[i]

    shape = list(map(int, args.shape.split(',')))
    dir = args.dir
    name = args.name.split(',')
    assert shape[0] == len(name)

    output = []

    for i in range(len(name)):
        row = []
        for j in range(shape[1]):
            img = cv2.imread('{}/{}{}.png'.format(dir, name[i], j))
            if row == []:
                row = img
            else:
                row = np.concatenate([row, img], axis=1)

        if output == []:
            output = row
        else:
            output = np.concatenate([output, row], axis=0)

    savename = dir + '/sum_img.png'
    cv2.imwrite(savename, output)


if __name__ = '__main__':
    args = get_argparser()
    concat(args)



