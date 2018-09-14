import cv2
import numpy as np

def write_boxline(input, mask, boxline):
    output = input + boxline
    for i, ch in enumerate([0, 151, 239]):
        output[:,:,i] = np.where(output[:,:,i] > 255, ch, output[:,:,i]) 
        
    return output.astype('uint8')


def create_boxline(mask, obj_rec, boxline, width):
    for rec in obj_rec:
        ul_y, ul_x, dr_y, dr_x = \
                rec[0], rec[1], rec[0] + rec[2], rec[1] + rec[3]
        boxline[ul_y-width:dr_y+width, ul_x-width:dr_x+width, :] = 255
    
    return boxline - mask

