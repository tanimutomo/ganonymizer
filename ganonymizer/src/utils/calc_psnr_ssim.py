import os
import cv2
import time
from skimage.measure import compare_ssim, compare_psnr


def main():
    data_dir = '../../data/videos/noon'
    summary_sheet = '../../data/docs'
    calc = Calcurator(data_dir, summary_sheet)
    calc.calcurate()
    calc.summary()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Calcurator:
    def __init__(self, data_dir, sheet):
        self.dir = data_dir
        self.doc_path = sheet
        self.count = 1
        self.start = time.time()

        self.psnr_in_out = AverageMeter()
        self.psnr_in_out_esp = AverageMeter()
        self.psnr_in_out_gfp = AverageMeter()
        self.ssim_in_out = AverageMeter()
        self.ssim_in_out_esp = AverageMeter()
        self.ssim_in_out_gfp = AverageMeter()
        
    def calcurate(self):
        for i in range(10):
            input = cv2.imread(os.path.join(self.dir, 'in_{}.png'.format(self.count)))
            out = cv2.imread(os.path.join(self.dir, 'out_{}.png'.format(self.count)))
            out_esp = cv2.imread(os.path.join(self.dir, 'out_esp_{}.png'.format(self.count)))
            out_gfp = cv2.imread(os.path.join(self.dir, 'out_gfp_{}.png'.format(self.count)))

            self.psnr_in_out.update(compare_psnr(input, out))
            self.psnr_in_out_esp.update(compare_psnr(input, out_esp))
            self.psnr_in_out_gfp.update(compare_psnr(input, out_gfp))
            self.ssim_in_out.update(compare_ssim(input, out))
            self.ssim_in_out_esp.update(compare_ssim(input, out_esp))
            self.ssim_in_out_gfp.update(compare_ssim(input, out_gfp))

            if count % 100 == 0:
                print('count: ', self.count)
                print('elapsed_time: ', time.time() - self.start)
                print('psnr_in_out: ', self.psnr_in_out.avg)
                print('psnr_in_out_esp: ', self.psnr_in_out_esp.avg)
                print('psnr_in_out_gfp: ', self.psnr_in_out_gfp.avg)
                print('ssim_in_out: ', self.ssim_in_out.avg)
                print('ssim_in_out_esp: ', self.ssim_in_out_esp.avg)
                print('ssim_in_out_gfp: ', self.ssim_in_out_gfp.avg)

            self.count += 1


    def summary(self):
        result = [
                '--SUMMARY about noon frames--',
                'Total images: {}'.format(self.count),
                'Elapsed Time: {}'.format(time.time() - self.start),
                'psnr_in_out: {}'.format(self.psnr_in_out.avg),
                'psnr_in_out_esp: {}'.format(self.psnr_in_out_esp.avg),
                'psnr_in_out_gfp: {}'.format(self.psnr_in_out_gfp.avg),
                'ssim_in_out: {}'.format(self.ssim_in_out.avg),
                'ssim_in_out_esp: {}'.format(self.ssim_in_out_esp.avg),
                'ssim_in_out_gfp: {}'.format(self.ssim_in_out_gfp.avg)
                ]

        with open(self.doc_path, mode='w') as f:
            f.write('\n'.join(result))

        with open(self.doc_path) as f:
            print(f.read())


if __name__ == '__main__':
    main()

