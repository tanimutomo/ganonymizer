import cv2
import numpy as np


def create_mask(size, mask_str):
    mask_shape = mask_str.split(',')
    mask_shape = list(map(int, mask_shape))
    ul_y, ul_x, dr_y, dr_x = mask_shape
    obj_rec = [ul_y, ul_x, dr_y - ul_y, dr_x - ul_x]

    mask = np.zeros(size).astype('uint8')
    mask[ul_y:dr_y, ul_x:dr_x, :] = np.ones((obj_rec[2], obj_rec[3], 3)).astype('uint8') * 255
    mask = mask.astype('uint8')

    return mask, [obj_rec]


def center_mask(size, mask_size):
    c_h, c_v = size[0]//2, size[1]//2
    base = mask_size // 2
    ul_y, ul_x, dr_y, dr_x = c_h - base, c_v - base, c_h + base, c_v + base
    obj_rec = [ul_y, ul_x, dr_y - ul_y, dr_x - ul_x]

    mask = np.zeros(size).astype('uint8')
    mask[ul_y:dr_y, ul_x:dr_x, :] = np.ones((obj_rec[2], obj_rec[3], 3)).astype('uint8') * 255
    mask = mask.astype('uint8')

    return mask, [obj_rec]


def edge_mask(size, mask_info):
    mask_info = mask_info.split(',')
    position = mask_info[0]
    distance = int(mask_info[1])
    mask_size = int(mask_info[2])

    c_h, c_v = size[0]//2, size[1]//2
    base = mask_size // 2
    
    if position == 'edge':
        ul_y, ul_x = c_h - base, size[1] - distance - mask_size
        dr_y, dr_x = c_h + base, size[1] - distance
    elif position == 'corner':
        dr_y, dr_x = size[0] - distance, size[1] - distance
        ul_y, ul_x = dr_y - mask_size, dr_x - mask_size
    else:
        raise RuntimeError('Invalid position')
        
    obj_rec = [ul_y, ul_x, dr_y - ul_y, dr_x - ul_x]

    mask = np.zeros(size).astype('uint8')
    mask[ul_y:dr_y, ul_x:dr_x, :] = np.ones((obj_rec[2], obj_rec[3], 3)).astype('uint8') * 255
    mask = mask.astype('uint8')

    return mask, [obj_rec]


def write_boxline(input, mask, boxline):
    output = input + boxline
    for i, ch in enumerate([0, 151, 239]):
        output[:,:,i] = np.where(output[:,:,i] > 255, ch, output[:,:,i]) 
        
    return output.astype('uint8')


def create_boxline(mask, obj_rec, width, original):
    boxline = np.zeros((original.shape))
    for rec in obj_rec:
        ul_y, ul_x, dr_y, dr_x = \
                rec[0], rec[1], rec[0] + rec[2], rec[1] + rec[3]
        boxline[ul_y-width:dr_y+width, ul_x-width:dr_x+width, :] = 255
    
    return boxline - mask


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    get_parser()
    create_mask()

# parser = argparse.ArgumentParser()
# parser.add_argument('--shape', type=list, default=[720, 1280, 3])
# parser.add_argument('--vertical', type=list, required=True)
# parser.add_argument('--horizontal', type=list, required=True)
# parser.add_argument('--output', type=str, required=True)
# args = parser.parse_args()

# shape = args.shape
# v = args.vertical
# h = args.horizontal
# print(shape, v, h)


# shape = [720, 1280, 3]
# end_y, end_x = 720, 1280
# height, width = 210, 210
# v = np.array([end_x - width, end_x])
# h = np.array([end_y - height, end_y])
# 
# move_v = 800
# move_h = 200
# 
# v = v - move_v
# h = h - move_h
# 
# mask = np.zeros(shape)
# # print(mask[h[0]:h[1], v[0]:v[1], :].shape)
# # print((v[1] - v[0], h[1] - h[0], shape[2]).shape)
# mask[h[0]:h[1], v[0]:v[1], :] = np.ones((height, width, shape[2])) * 255
# 
# cv2.imwrite('./ex_images/m_v{}-{}_h{}-{}.png'.format(v[0], v[1], h[0], h[1]), mask)
