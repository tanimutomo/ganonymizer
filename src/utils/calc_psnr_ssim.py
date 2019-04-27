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
        for i in range(1, 3200):
            input = cv2.imread(os.path.join(self.dir, 'input', '{}.png'.format(i)))
            out_esp_off = cv2.imread(os.path.join(self.dir, 'out_esp_off', '{}.png'.format(i)))
            out_gfp_off = cv2.imread(os.path.join(self.dir, 'out_gfp_off', '{}.png'.format(i)))
            out_esp_on = cv2.imread(os.path.join(self.dir, 'out_esp_on', '{}.png'.format(i)))
            out_gfp_on = cv2.imread(os.path.join(self.dir, 'out_gfp_on', '{}.png'.format(i)))

            self.psnr_esp_off.update(compare_psnr(input, out_esp_off))
            self.psnr_esp_on.update(compare_psnr(input, out_esp_on))
            self.psnr_gfp_off.update(compare_psnr(input, out_gfp_off))
            self.psnr_gfp_on.update(compare_psnr(input, out_gfp_on))

            self.ssim_esp_off.update(compare_ssim(input, out_esp_off, multichannel=True))
            self.ssim_esp_on.update(compare_ssim(input, out_esp_on, multichannel=True))
            self.ssim_gfp_off.update(compare_ssim(input, out_gfp_off, multichannel=True))
            self.ssim_gfp_on.update(compare_ssim(input, out_gfp_on, multichannel=True))

            if i % 100 == 0:
                print('----------------------SUMMARY----------------------')
                print('count: ', i)
                print('elapsed_time: ', time.time() - self.start)

                print('PSNR_ESP_OFF: ', self.psnr_esp_off.avg)
                print('PSNR_ESP_ON: ', self.psnr_esp_on.avg)
                print('PSNR_GFP_OFF: ', self.psnr_gfp_off.avg)
                print('PSNR_GFP_ON: ', self.psnr_gfp_on.avg)

                print('SSIM_ESP_OFF: ', self.ssim_esp_off.avg)
                print('SSIM_ESP_ON: ', self.ssim_esp_on.avg)
                print('SSIM_GFP_OFF: ', self.ssim_gfp_off.avg)
                print('SSIM_GFP_ON: ', self.ssim_gfp_on.avg)


    def summary(self):
        result = [
                '--SUMMARY about noon frames--',
                'Total images: {}'.format(3199),
                'Elapsed Time: {}'.format(time.time() - self.start),
                'PSNR_ESP_OFF: {}'.format(self.psnr_esp_off.avg),
                'PSNR_ESP_ON: {}'.format(self.psnr_esp_on.avg),
                'PSNR_GFP_OFF: {}'.format(self.psnr_gfp_off.avg),
                'PSNR_GFP_ON: {}'.format(self.psnr_gfp_on.avg),
                'SSIM_ESP_OFF: {}'.format(self.ssim_esp_off.avg),
                'SSIM_ESP_ON: {}'.format(self.ssim_esp_on.avg),
                'SSIM_GFP_OFF: {}'.format(self.ssim_gfp_off.avg),
                'SSIM_GFP_ON: {}'.format(self.ssim_gfp_on.avg)
                ]

        with open(self.doc_path, mode='w') as f:
            f.write('\n'.join(result))

        with open(self.doc_path) as f:
            print(f.read())


if __name__ == '__main__':
    main()

