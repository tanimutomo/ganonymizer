import numpy as np
import cv2

def pre_padding(input, mask, vi, hi, origin):
    s_v, e_v, s_h, e_h = vi.min(), vi.max(), hi.min(), hi.max()
    v_l = [origin[1]-4, origin[1]-3, origin[1]-2, origin[1]-1, s_v-1, s_v-2, s_v-3, s_v-4]
    h_l = [origin[0]-4, origin[0]-3, origin[0]-2, origin[0]-1, s_h-1, s_h-2, s_h-3, s_h-4]
    if e_v > origin[1] - 5:
        pv1 = input[:, v_l[v_l.index(e_v)+1], :].reshape(-1, 1, 3)
        pv2 = input[:, v_l[v_l.index(e_v)+2], :].reshape(-1, 1, 3)
        pv3 = input[:, v_l[v_l.index(e_v)+3], :].reshape(-1, 1, 3)
        pv4 = input[:, v_l[v_l.index(e_v)+4], :].reshape(-1, 1, 3)
        pv = np.concatenate([pv1, pv2, pv3, pv4], axis=1)
        input = np.concatenate([input, pv], axis=1)
        mask = np.concatenate([mask, np.zeros(pv.shape, dtype='uint8')], axis=1)

    if e_h > origin[0] - 5:
        ph1 = input[h_l[h_l.index(e_h)+1], :, :].reshape(1, -1, 3)
        ph2 = input[h_l[h_l.index(e_h)+2], :, :].reshape(1, -1, 3)
        ph3 = input[h_l[h_l.index(e_h)+3], :, :].reshape(1, -1, 3)
        ph4 = input[h_l[h_l.index(e_h)+4], :, :].reshape(1, -1, 3)
        ph = np.concatenate([ph1, ph2, ph3, ph4], axis=0)
        input = np.concatenate([input, ph], axis=0)
        mask = np.concatenate([mask, np.zeros(ph.shape, dtype='uint8')], axis=0)

    return input, mask

def cut_padding(out, origin):
    if out.shape[0] != origin[0]:
        out = np.delete(out, [origin[0], origin[0]+1, origin[0]+2, origin[0]+3], axis=0)

    if out.shape[1] != origin[1]:
        out = np.delete(out, [origin[1], origin[1]+1, origin[1]+2, origin[1]+3], axis=1)

    return out

def detect_large_mask(mask, thresh):
    idx, col = np.where(mask[:, :, 0]!=0)
    seq_h, seq_w = [], []
    rec = []
    idx_set = list(sorted(set(idx)))
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

def check_rec_position(mask, out1, out2, ip, length, valid):
    if np.sum(mask[out1[0]:out1[2], out1[1]:out1[3], 0] > 0) > ip * length / 8:
        out1 = out2
    if np.sum(mask[out2[0]:out2[2], out2[1]:out2[3], 0] > 0) > ip * length / 8:
        if np.all(out1 == out2):
            valid = False
        else:
            out2 = out1
    
    return out1, out2, valid

def arange_rec_out(rec_out, opp_rec, ip, ip_axis):
    if rec_out.shape[ip_axis] >= ip / 8 and rec_out.shape[ip_axis] < ip:
        if ip_axis == 0:
            rec_out = cv2.resize(rec_out, (rec_out.shape[ip_axis+1], ip))
        if ip_axis == 1:
            rec_out = cv2.resize(rec_out, (ip, rec_out.shape[ip_axis-1]))
    elif rec_out.shape[ip_axis] < ip / 8:
        rec_out = opp_rec
        
    return rec_out

def grid_interpolation(input, mask, rec, rec_w):
    if rec != []:
        for r in rec:
            print('[INFO] detect large mask which (y, x, h, w) is {}'.format(r))
            valid_lr, valid_ud = True, True
            y, x, h, w = r
            g_h = int(h / 8)
            g_w = int(w / 8)
            i_h = int(h / rec_w)
            i_w = int(w / rec_w)

            # recのsy, sx, ey, ex
            out_l = [y, x-i_w, y+h, x]
            out_r = [y, x+w, y+h, x+w+i_w]
            out_u = [y-i_h, x, y, x+w]
            out_d = [y+h, x, y+h+i_h, x+w]
            
            out_l, out_r, valid_lr = check_rec_position(mask, out_l, out_r, i_w, h, valid_lr)
            out_u, out_d, valid_ud = check_rec_position(mask, out_u, out_d, i_h, w, valid_ud)
            
            if valid_lr:
                rec_out_l = input[out_l[0]:out_l[2], out_l[1]:out_l[3], :]
                rec_out_r = input[out_r[0]:out_r[2], out_r[1]:out_r[3], :]
                rec_out_l = arange_rec_out(rec_out_l, rec_out_r, i_w, 1)
                rec_out_r = arange_rec_out(rec_out_r, rec_out_l, i_w, 1)

                input[y : y+h, x+2*g_w : x+2*g_w+i_w, :] = (2 * rec_out_l + 1 * rec_out_r) / 3
                input[y : y+h, x+w-3*g_w : x+w-3*g_w+i_w, :] = (1 * rec_out_l + 2 * rec_out_r) / 3
                # input[y : y+h, x+2*g_w : x+2*g_w+i_w, :] = rec_out_l
                # input[y : y+h, x+w-3*g_w : x+w-3*g_w+i_w, :] = rec_out_r
                mask[y : y+h, x+2*g_w : x+2*g_w+i_w, :] = 0
                mask[y : y+h, x+w-3*g_w : x+w-3*g_w+i_w, :] = 0

            if valid_ud:
                rec_out_u = input[out_u[0]:out_u[2], out_u[1]:out_u[3], :]
                rec_out_d = input[out_d[0]:out_d[2], out_d[1]:out_d[3], :]
                rec_out_u = arange_rec_out(rec_out_u, rec_out_d, i_h, 0)
                rec_out_d = arange_rec_out(rec_out_d, rec_out_u, i_h, 0)
                print(rec_out_d.shape)

                input[y+2*g_h : y+2*g_h+i_h, x : x+w, :] = (2 * rec_out_u + 1 * rec_out_d) / 3
                input[y+h-3*g_h : y+h-3*g_h+i_h, x : x+w, :] = (1 * rec_out_u + 2 * rec_out_d) / 3
                # input[y+2*g_h : y+2*g_h+i_h, x : x+w, :] = rec_out_u
                # input[y+h-3*g_h : y+h-3*g_h+i_h, x : x+w, :] = rec_out_d
                mask[y+2*g_h : y+2*g_h+i_h, x : x+w, :] = 0
                mask[y+h-3*g_h : y+h-3*g_h+i_h, x : x+w, :] = 0

            # rec_mean = (rec_out_l + rec_out_r + rec_out_u.transpose(1, 0, 2) + rec_out_d.transpose(1, 0, 2)) / 4
            # print(rec_mean.shape)
            # rec_mean = cv2.resize(rec_mean, (2*g_h, 2*g_w))
            # print(rec_mean.shape)
            # print(input[y+3*g_h : y+5*g_h, x+3*g_w : x+5*g_w, :].shape)
            # input[y+3*g_h : y+5*g_h, x+3*g_w : x+5*g_w, :] = rec_mean
            # mask[y+3*g_h : y+5*g_h, x+3*g_w : x+5*g_w, :] = 0


    return input, mask

def sparse_patch(input, out256, mask, rec, comp_size):
    for r in rec:
        y, x, h, w = r
        # input[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :] = out256[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :]
        # mask[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :] = 0

        # interval = 60
        inter_h, inter_w = int(h / 4), int(w / 4)
        # inter_h = int(input.shape[0] / comp_size[0])
        # inter_w = int(input.shape[1] / comp_size[1])
        # print('[INFO] interval of interpolation is: {}'.format(interval))
        print('[INFO] interval of interpolation is: (h: {}, w: {})'.format(inter_h, inter_w))

        py = y + inter_h
        patch_size = int(h / 8)
        input[y+inter_h:y+inter_h+patch_size, x:x+w, :] = out256[y+inter_h:y+inter_h+patch_size, x:x+w, :]
        mask[y+inter_h:y+inter_h+patch_size, x:x+w, :] = 0
        input[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = out256[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :]
        mask[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = 0
        input[y:y+h, x+inter_w:x+inter_w+patch_size, :] = out256[y:y+h, x+inter_w:x+inter_w+patch_size, :]
        mask[y:y+h, x+inter_w:x+inter_w+patch_size, :] = 0
        input[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :] = out256[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :]
        mask[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :] = 0

        # while py + patch_size < y + h:
        #     input[y+inter_h:y+inter_h+patch_size, x:x+w, :] = out256[y+inter_h:y+inter_h+patch_size, x:x+w, :]
        #     mask[y+inter_h:y+inter_h+patch_size, x:x+w, :] = 0
        #     input[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = out256[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :]
        #     mask[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = 0
        #     py += inter_h
        #     py += 2 * inter_h

        # px = x + inter_w
        # while px + patch_size < x + w:
        #     input[y:y+h, px:px+patch_size, :] = out256[y:y+h, px:px+patch_size, :]
        #     mask[y:y+h, px:px+patch_size, :] = 0
        #     px += inter_w
            # px += 2 * inter_w

        # while py + patch_size < y + h:
        #     if indent:
        #         px = x + patch_size
        #     else:
        #         px = x

        #     while px + patch_size < x + w:
        #         input[py:py+patch_size, px:px+patch_size, :] = out256[py:py+patch_size, px:px+patch_size, :]
        #         mask[py:py+patch_size, px:px+patch_size, :] = 0
        #         # input[py, px, :] = out256[py, px, :]
        #         # mask[py, px, :] = 0
        #         # input[py:py+inter_h, px:px+inter_w, :] = out256[py:py+inter_h, px:px+inter_w, :]
        #         # mask[py:py+inter_h, px:px+inter_w, :] = 0
        #         px += inter_w
        #     py += inter_h
        #     if indent:
        #         indent = False
        #     else:
        #         indent = True
    
    return input, mask

if __name__ == '__main__':
    input = (np.random.rand(40, 20, 3) * 10).astype('uint8')

    mask = np.zeros((40, 20, 3)).astype('uint8')
    m = np.ones((8, 16, 3))*3
    m2 = np.ones((25, 8, 3))*3
    mask[3:11, 3:19, :] = m
    mask[10:35, 8:16, :] = m2

    print(input.shape)
    print(mask.shape)

    thresh = 6
    rec_w = 8
    n_input = input.copy()
    n_mask = mask.copy()
    out256 = np.zeros(input.shape)
    rec = detect_large_mask(mask, thresh)
    n_input, n_mask = sparse_patch(n_input, out256, n_mask, rec, [8, 6])
    # n_input, n_mask = grid_interpolation(n_input, n_mask, rec, rec_w)

    for i in range(1):
        print(input[:, :, i]==n_input[:, :, i])
        print(n_mask[:, :, i])

