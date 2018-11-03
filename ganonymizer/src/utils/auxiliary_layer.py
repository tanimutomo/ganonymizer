import numpy as np
import cv2


def calc_sml_size(largs_thresh, origin, max):
    ratio = large_thresh / max
    sml = int(np.floor(origin * ratio))
    sml_do2 = sml % 100
    sml_up = sml - sml_do2
    sml_do2 = (sml_do2 // 4) * 4
    sml = sml_up + sml_do2

    return sml

def pre_padding(input, mask, thresh, wi, hi, size, flag):
    s_w, e_w, s_h, e_h = wi.min(), wi.max(), hi.min(), hi.max()

    if (e_h > size[0] - thresh+1 and s_h < thresh) or (e_w > size[1] - thresh+1 and s_w < thresh):
        print('[INFO] Not enable to pre_padding because Mask is too large')
        return input, mask, flag
    else:
        if e_h - (size[0] - 1) < thresh:
            input, mask = prepad_new(input, mask, e_h, s_h, 0, thresh)
            flag['hd'] = True

        if e_w - (size[1] - 1) < thresh:
            input, mask = prepad_new(input, mask, e_w, s_w, 1, thresh)
            flag['wr'] = True

        if s_h < thresh:
            input, mask, prepad_new(input, mask, s_h, e_h, 0, thresh)
            flag['hu'] = True

        if s_w < thresh:
            input, mask = prepad_new(input, mask, s_w, e_w, 1, thresh)
            flag['wl'] = True

        return input, mask, flag


#     base = np.arange(0, thresh)
#     hu_l = list(np.concatenate([base[::-1], base+1+e_h], axis=0))
#     wl_l = list(np.concatenate([base[::-1], base+1+e_w], axis=0))
#     hd_l = list(np.concatenate([origin[0]-base[::-1]-1, s_h-1-base], axis=0))
#     wr_l = list(np.concatenate([origin[1]-base[::-1]-1, s_w-1-base], axis=0))
# 
#         if e_h > origin[0] - thresh+1:
#             input, mask, flag = prepad(input, mask, flag, 'hd', hd_l, e_h)
# 
#         if e_w > origin[1] - thresh+1:
#             input, mask, flag = prepad(input, mask, flag, 'wr', wr_l, e_w)
# 
#         if s_h < thresh:
#             input, mask, flag = prepad(input, mask, flag, 'hu', hu_l, s_h)
# 
#         if s_w < thresh:
#             input, mask, flag = prepad(input, mask, flag, 'wl', wl_l, s_w)


def cut_padding(out, origin, flag):
    if flag['hu']:
        out = np.delete(out, [0, 1, 2, 3], axis=0)
        flag['hu'] = False

    if flag['wl']:
        out = np.delete(out, [0, 1, 2, 3], axis=1)
        flag['wl'] = False

    if flag['hd']:
        out = np.delete(out, [origin[0], origin[0]+1, origin[0]+2, origin[0]+3], axis=0)
        flag['hd'] = False

    if flag['wr']:
        out = np.delete(out, [origin[1], origin[1]+1, origin[1]+2, origin[1]+3], axis=1)
        flag['wr'] = False

    return out


def pseudo_mask_division(input, out_sml, mask, rec, thresh):
    for r in rec:
        y, x, h, w = r
        # input[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :] = out256[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :]
        # mask[y+int(h/4):y+int(h*3/4), x+int(w/4):x+int(w*3/4), :] = 0

        # interval = 60
        inter_h, inter_w = int(h / 4), int(w / 4)
        # inter_h = int(input.shape[0] / comp_size[0])
        # inter_w = int(input.shape[1] / comp_size[1])
        # print('[INFO] interval of interpolation is: {}'.format(interval))
        if w > thresh and h > thresh:
            print('[INFO] Devided small mask size: (h: {}, w: {})'.format(inter_h, inter_w))
            # print(r)
            # raise RuntimeError('Include Large Mask')
            # print('rec: ', y, x, h, w)

            py = y + inter_h
            patch_size = int(h / 8)
            if h > thresh:
                # print('input.shape: {}'.format(input.shape))
                # print('out256.shape: {}'.format(out256.shape))
                input[y+inter_h:y+inter_h+patch_size, x:x+w, :] = out_sml[y+inter_h:y+inter_h+patch_size, x:x+w, :]
                mask[y+inter_h:y+inter_h+patch_size, x:x+w, :] = 0
                input[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = out_sml[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :]
                mask[y+h-inter_h-patch_size:y+h-inter_h, x:x+w, :] = 0
                # print('vertical_up_interbar: ', y+inter_h, x, (y+inter_h+patch_size)-(y+inter_h), w)
                # print('vertical_down_interbar: ', y+h-inter_h-patch_size, x, y+h-inter_h-y+h-inter_h-patch_size, w)

            if w > thresh:
                input[y:y+h, x+inter_w:x+inter_w+patch_size, :] = out_sml[y:y+h, x+inter_w:x+inter_w+patch_size, :]
                mask[y:y+h, x+inter_w:x+inter_w+patch_size, :] = 0
                input[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :] = out_sml[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :]
                mask[y:y+h, x+w-inter_w-patch_size:x+w-inter_w, :] = 0
                # print('horizontal_left_interbar: ', y, x+inter_w, h, (x+inter_w+patch_size)-(x+inter_w))
                # print('horizontal_right_interbar: ', y, x+w-inter_w-patch_size, h, x+w-inter_w-x+w-inter_w-patch_size)

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


# following function is not used (2018/11/02)

def prepad_new(input, mask, edg, opp, direction, thresh):
    begin = 0
    end = input.shape[direction] - 1
    if direction == 0:
        if edg > opp:
            if end == edg:
                pad = input[opp-1, :, :].reshape(1, -1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

            elif end - edg < thresh:
                pad = input[end, :, :].reshape(1, -1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)
        
        elif edg < opp:
            if begin == edg:
                pad = input[opp+1, :, :].reshape(1, -1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

            elif edg - begin < thresh:
                pad = input[begin, :, :].reshape(1, -1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

    elif direction == 1:
        if edg > opp:
            if end == edg:
                pad = input[:, opp-1, :].reshape(-1, 1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

            elif end - edg < thresh:
                pad = input[:, end, :].reshape(-1, 1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)
        
        elif edg < opp:
            if begin == edg:
                pad = input[:, opp+1, :].reshape(-1, 1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

            elif edg - begin < thresh:
                pad = input[:, begin, :].reshape(-1, 1, 3)
                input, mask = concat_pad(input, mask, pad, direction, thresh)

    return input.astype('uint8'), mask.astype('uint8')


def concat_pad(input, mask, pad, axis, iter):
    mpad = np.zeros(pad.shape, dtype='uint8')
    for i in range(4):
        input = np.concatenate([input, pad], axis=axis)
        mask = np.concatenate([mask, mpad], axis=axis)

    return input, mask


# ph1 = input[hd_l[hd_l.index(e_h)+1], :, :].reshape(1, -1, 3)
# ph2 = input[hd_l[hd_l.index(e_h)+2], :, :].reshape(1, -1, 3)
# ph3 = input[hd_l[hd_l.index(e_h)+3], :, :].reshape(1, -1, 3)
# ph4 = input[hd_l[hd_l.index(e_h)+4], :, :].reshape(1, -1, 3)
# ph = np.concatenate([ph1, ph2, ph3, ph4], axis=0)
# input = np.concatenate([input, ph], axis=0)
# mask = np.concatenate([mask, np.zeros(ph.shape, dtype='uint8')], axis=0)
# flag['hd'] = True

def prepad(input, mask, flag, f_name, list, line):
    for i in range(thresh):
        pad_raw = input[list[list.index(line)+i+1], :, :].reshape(1, -1, 3)
        pad_raws.append(pad_raw)

    prepad = np.concatenate(pad_raws, axis=0)
    input = np.concatenate([input, prepad], axis=0)
    mask = np.concatenate([mask, np.zeros(prepad.shape, dtype='uint8')], axis=0)
    flag[f_name] = True

    return input, mask, flag


def detect_large_mask(mask):
    idx, col = np.where(mask[:, :, 0]!=0)
    seq_h, seq_w = [], []
    rec = []
    # idx_setは縦に並んでいる数が高さになる．厳密にはマスクがある場所のyのindex
    idx_set = list(sorted(set(idx)))
    for i in idx_set:
        # 全マスクのyのindexに対して，いくつそのyがidxの方に並んでいるかをみて，xのindexが閾値以上並んでいる場合は，そのyのindexとxがいくつ並んでいるかを追加する．
        # if np.sum(idx == i) >= thresh:
        seq_h.append([i, np.sum(idx == i)])

    if seq_h != []:
        # print('seq_h: {}'.format(np.array(seq_h)))
        # print('seq_h.shape: {}'.format(np.array(seq_h).shape))
        seq_h = np.array(seq_h)
        seq_h_set = np.array(list(set(seq_h[:, 1])))
        # print('seq_h_set: {}'.format(np.array(seq_h_set)))
        # print('seq_h_set.shape: {}'.format(np.array(seq_h_set).shape))

        seq_h_div = []
        for i in seq_h_set:
            seq_h_div.append(seq_h[seq_h[:, 1] == i])
        # print('seq_h_div: {}'.format(np.array(seq_h_div)))
        # print('seq_h_div.shape: {}'.format(np.array(seq_h_div).shape))

        seq_h_div = np.array(seq_h_div)
        for i in seq_h_div:
            s_h, e_h = i[0, 0], i[-1, 0]
            tmp_rec = np.arange(s_h, e_h+1)
            # if len(tmp_rec) >= thresh:
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

