import cv2
import time
import subprocess


def main():
    demo = RealTimeDemo()
    demo.capture()
    demo.display(True)
    demo.send()
    demo.execute()
    demo.get()
    demo.display(False)


class RealTimeDemo:
    def __init__(self):
        self.local_img = './ganonymizer/data/images/tmp1.png'
        self.remote_img = 'chasca:Project/ganonymizer/ganonymizer/data/images/tmp1.png'
        self.local_out = './ganonymizer/data/images/out0_tmp1.png'
        self.remote_out = 'chasca:Project/ganonymizer/ganonymizer/data/images/out0_tmp1.png'

    def capture(self):
        print('capture')
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
        print('send')
        cmd = ['scp', 
                self.local_img,
                self.remote_img]
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def get(self):
        print('get')
        cmd = ['scp', 
                self.remote_out,
                self.local_out]
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def execute(self):
        print('execute')
        cmd = ['kronos', 
                'job',
                '--m',
                'ganonymizer.main']
        runcmd = subprocess.check_call(cmd)
        print (runcmd)

    def display(self, img):
        print('display')
        window = "Push ESC key to stop this program"
        if img:
            img = cv2.imread(self.local_img)
        else:
            img = cv2.imread(self.local_out)
        cv2.imshow(window, img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
