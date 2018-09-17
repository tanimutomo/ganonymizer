import cv2
import numpy as np

infile = '../data/video/inter10_noon.avi'
video = []

cap = cv2.VideoCapture(infile)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(count, fps, W, H))

count = 1
n_w, n_h = int(W / 4), int(H / 4)
n_w_d2 = n_w % 100
n_w_up = n_w - n_w_d2
n_w_d2 = n_w_d2 - (n_w_d2 % 4)
n_w = n_w_up + n_w_d2

n_h_d2 = n_h % 100
n_h_up = n_h - n_h_d2
n_h_d2 = n_h_d2 - (n_h_d2 % 4)
n_h = n_h_up + n_h_d2

#動画のコーディックを指定するコード
fourcc = cv2.VideoWriter_fourcc(*'XVID')
outfile = '../data/video/ex_small4_{}'.format(infile.split('/')[-1])
fps = 5
#動画像の書き込み仕様設定（動画の再生速度と解像度を指定）
writer = cv2.VideoWriter(outfile, fourcc, fps, (int(n_w),int(n_h)))
#ガンマテーブル作成
# gamma_cvt = np.zeros((256,1),dtype = 'uint8')
# for i in range(256):
#     gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
# frame = cv2.LUT(frame,gamma_cvt)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if count >= 100 and count <= 105:
            frame = cv2.resize(frame, (n_w, n_h))
            # cv2.imshow('frame', frame)
            # video.append(frame)
            writer.write(frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    else:
        break
    count += 1
    if count >= 150:
        break

cap.release()
writer.release()

# video = cv2.convertTo(video, CV_8U, 255)

# print(np.array([video]).shape)
# frames, height, width, ch = np.array([video]).shape

# fourcc = cv2.VideoWriter_fourcc(*codecs)
# writer = cv2.VideoWriter(outfile, fourcc, fps, (n_w, n_h))
# for i in range(frames):
#     writer.write(video[i])
# writer.release()
