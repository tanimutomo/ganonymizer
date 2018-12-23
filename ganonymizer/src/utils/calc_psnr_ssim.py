import os
import cv2
import time
from skimage.measure import compare_ssim, compare_psnr


def main():
    data_dir = 'ganonymizer/data/videos/noon'
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


class PSNRSSIMCalcurator:
    def __init__(self, data_dir):
        self.dir = data_dir
        self.doc_path = os.path.join(data_dir, 'summary.txt')
        self.count = 1
        self.start = time.time()

        self.psnr_esp_off = AverageMeter()
        self.psnr_esp_on = AverageMeter()
        self.psnr_gfp_off = AverageMeter()
        self.psnr_gfp_on = AverageMeter()

        self.ssim_esp_off = AverageMeter()
        self.ssim_esp_on = AverageMeter()
        self.ssim_gfp_off = AverageMeter()
        self.ssim_gfp_on = AverageMeter()

        
    def calcurate(self):
        for i in range(10):
            input = cv2.imread(os.path.join(self.dir, 'input', '{}.png'.format(self.count)))
            out_esp_off = cv2.imread(os.path.join(self.dir, 'out_esp_off', '{}.png'.format(self.count)))
            out_gfp_off = cv2.imread(os.path.join(self.dir, 'out_gfp_off', '{}.png'.format(self.count)))
            out_esp_on = cv2.imread(os.path.join(self.dir, 'out_esp_on', '{}.png'.format(self.count)))
            out_gfp_on = cv2.imread(os.path.join(self.dir, 'out_gfp_on', '{}.png'.format(self.count)))

            self.psnr_esp_off.update(compare_psnr(input, out_esp_off))
            self.psnr_esp_on.update(compare_psnr(input, out_esp_on))
            self.psnr_gfp_off.update(compare_psnr(input, out_gfp_off))
            self.psnr_gfp_on.update(compare_psnr(input, out_gfp_on))

            self.ssim_esp_off.update(compare_ssim(input, out_esp_off, multichannel=True))
            self.ssim_esp_on.update(compare_ssim(input, out_esp_on, multichannel=True))
            self.ssim_gfp_off.update(compare_ssim(input, out_gfp_off, multichannel=True))
            self.ssim_gfp_on.update(compare_ssim(input, out_gfp_on, multichannel=True))

            if self.count % 10 == 0:
                print('count: ', self.count)
                print('elapsed_time: ', time.time() - self.start)

                print('psnr_esp_off: ', self.psnr_esp_off.avg)
                print('psnr_esp_on: ', self.psnr_esp_on.avg)
                print('psnr_gfp_off: ', self.psnr_gfp_off.avg)
                print('psnr_gfp_on: ', self.psnr_gfp_on.avg)

                print('ssim_esp_off: ', self.ssim_esp_off.avg)
                print('ssim_esp_on: ', self.ssim_esp_on.avg)
                print('ssim_gfp_off: ', self.ssim_gfp_off.avg)
                print('ssim_gfp_on: ', self.ssim_gfp_on.avg)

            self.count += 1


    def summary(self):
        result = [
                '--SUMMARY about noon frames--',
                'Total images: {}'.format(self.count),
                'Elapsed Time: {}'.format(time.time() - self.start),
                'psnr_esp_off: {}'.format(self.psnr_esp_off.avg),
                'psnr_esp_on: {}'.format(self.psnr_esp_on.avg),
                'psnr_gfp_off: {}'.format(self.psnr_gfp_off.avg),
                'psnr_gfp_on: {}'.format(self.psnr_gfp_on.avg),
                'ssim_esp_off: {}'.format(self.ssim_esp_off.avg),
                'ssim_esp_on: {}'.format(self.ssim_esp_on.avg),
                'ssim_gfp_off: {}'.format(self.ssim_gfp_off.avg),
                'ssim_gfp_on: {}'.format(self.ssim_gfp_on.avg)
                ]

        with open(self.doc_path, mode='w') as f:
            f.write('\n'.join(result))

        with open(self.doc_path) as f:
            print(f.read())


if __name__ == '__main__':
    main()

