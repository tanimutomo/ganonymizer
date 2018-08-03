import numpy as np
import cv2

def detect_large_mask(mask):
    thresh = 2
    count = 1
    idx, col = np.where(mask!=0)
    seq_h, seq_w = [], []
    rec = []
    idx_set = list(set(idx))
    for i in idx_set:
        if np.sum(idx == i) >= thresh:
            seq_h.append([i, np.sum(idx == i)])

    if seq_h != []:
        seq_h = np.array(seq_h)
        seq_h_set = np.array(list(set(seq_h[:, 1])))

        seq_h_div = []
        for i in seq_h_set:
            seq_h_div.append(seq_h[seq_h[:, 1] == i])

        seq_h_div = np.array(seq_h_div)
        for i in seq_h_div:
            s_h, e_h = i[0, 0], i[-1, 0]
            tmp_rec = np.arange(s_h, e_h+1)
            if len(tmp_rec) >= thresh:
                if np.all(i[:, 0] == tmp_rec):
                    tmp_idx, tmp_col = np.where([idx==s_h])
                    s_v = col[tmp_col.min()]
                    rec.append(np.array([s_h, s_v, len(tmp_rec), i[0, 1]]))

    return np.array(rec)

def check_rec_position(mask, out1, out2, grid, length, valid):
    if np.sum(mask[out1[0]:out1[2], out1[1]:out1[3]] > 0) > grid * length / 8:
        out1 = out2
    if np.sum(mask[out2[0]:out2[2], out2[1]:out2[3]] > 0) > grid * length / 8:
        if np.all(out1 == out2):
            valid = False
        else:
            out2 = out1
    
    return out1, out2, valid

def arange_rec_out(rec_out, opp_rec, grid):
    if rec_out.shape[1] >= grid / 8 and rec_out.shape[1] < grid:
        rec_out = cv2.resize(rec_out, (grid, rec_out.shape[0]))
    elif rec_out.shape[1] < grid / 8:
        rec_out = opp_rec
        
    return rec_out

def pre_grid_support(input, mask, rec):
    if rec != []:
        n_input = input.copy()
        for r in rec:
            print('[INFO] detect large mask which (y, x, h, w) is {}'.format(r))
            valid_lr, valid_ud = True, True
            y, x, h, w = r
            g_h = int(h / 8)
            g_w = int(w / 8)
            # print('g_h: {}, g_w: {}'.format(g_h, g_w))
#             print(x, g_w)
#             print(x-g_w, x)
#             print(x+w, x+w+g_w)

            out_l = [y, x-g_w, y+h, x]
            out_r = [y, x+w, y+h, x+w+g_w]
            out_u = [y-g_h, x, y, x+w]
            out_d = [y+h, x, y+h+g_h, x+w]
            
            out_l, out_r, valid_lr = check_rec_position(mask, out_l, out_r, g_w, h, valid_lr)
            out_u, out_d, valid_ud = check_rec_position(mask, out_u, out_d, g_h, w, valid_ud)
            
            if valid_lr:
                rec_out_l = n_input[out_l[0]:out_l[2], out_l[1]:out_l[3]]
                rec_out_r = n_input[out_r[0]:out_r[2], out_r[1]:out_r[3]]
                rec_out_l = arange_rec_out(rec_out_l, rec_out_r, g_w)
                rec_out_r = arange_rec_out(rec_out_r, rec_out_l, g_w)
#                 print('rec_out_l', rec_out_l)
#                 print('rec_out_r', rec_out_r)

                n_input[y : y+h, x+2*g_w : x+3*g_w] = (2 * rec_out_l + 1 * rec_out_r) / 3
                n_input[y : y+h, x+w-3*g_w : x+w-2*g_w] = (1 * rec_out_l + 2 * rec_out_r) / 3
                mask[y : y+h, x+2*g_w : x+3*g_w] = 0
                mask[y : y+h, x+w-3*g_w : x+w-2*g_w] = 0

            if valid_ud:
                rec_out_u = n_input[out_u[0]:out_u[2], out_u[1]:out_u[3]]
                rec_out_d = n_input[out_d[0]:out_d[2], out_d[1]:out_d[3]]
                # print('rec_out_u', rec_out_u.shape)
                # print('rec_out_d', rec_out_d.shape)
                rec_out_u = arange_rec_out(rec_out_u, rec_out_d, g_h)
                rec_out_d = arange_rec_out(rec_out_d, rec_out_u, g_h)
                # print('rec_out_u', rec_out_u.shape)
                # print('rec_out_d', rec_out_d.shape)

                n_input[y+2*g_h : y+3*g_h, x : x+w] = (2 * rec_out_u + 1 * rec_out_d) / 3
                n_input[y+h-3*g_h : y+h-2*g_h, x : x+w] = (1 * rec_out_u + 2 * rec_out_d) / 3
                mask[y+2*g_h : y+3*g_h, x : x+w] = 0
                mask[y+h-3*g_h : y+h-2*g_h, x : x+w] = 0


    return n_input, mask


if __name__ == '__main__':
    input = (np.random.rand(40, 20) * 10).astype('uint8')

    mask = np.zeros((40, 20)).astype('uint8')
    m = np.ones((8, 16))*3
    m2 = np.ones((25, 8))*3
    mask[3:11, 3:19] = m
    mask[10:35, 8:16] = m2

    rec = detect_large_mask(mask)
    n_input, n_mask = pre_grid_support(input, mask, rec)

    print(input==n_input)
    print(n_mask)

