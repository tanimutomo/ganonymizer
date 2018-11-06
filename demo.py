import cv2
import time
import subprocess


def main():
    demo = RealTimeDemo()
    demo.capture()
    demo.send()
    demo.execute()
    time.sleep(3)
    demo.get()
    demo.display()


class RealTimeDemo:
    def __init__(self):
        self.local_img = './ganonymizer/data/images/tmp1.png'
        self.remote_img = 'chasca:Project/ganonymizer/ganonymizer/data/images/tmp1.png'
        self.local_out = './ganonymizer/data/images/out0_tmp1.png'
        self.remote_out = 'chasca:Project/ganonymizer/ganonymizer/data/out0_images/tmp1.png'

    def capture(self):
        capture = cv2.VideoCapture(0)

        # キャプチャ処理
        while(True):
            # key = cv2.waitKey(5)
            # if(key == 27):
            #     print("exit.")
            #     break

            # 画像キャプチャ
            ret, frame = capture.read()

            # 取り込み開始になっていなかったら上の処理に戻る
            if ret:
                cv2.imwrite(self.local_img, frame)
            else:
                print('Capture Error')
                break

            break

        capture.release()
        cv2.destroyAllWindows()

    def send(self):
        cmd = ['scp', 
                self.local_img,
                self.remote_img]
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def get(self):
        cmd = ['scp', 
                self.remote_out,
                self.local_out]
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def execute(self):
        cmd = ['kronos', 
                'job',
                '--m',
                'ganonymizer.main']
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def display_img(self):
        window = "Push ESC key to stop this program"
        output = cv2.imread(self.local_out)
        cv2.imshow(window, output)
        key = cv2.waitKey(5)
        if(key == 27):
            print("exit.")
            break
