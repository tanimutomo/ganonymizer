# GANonymizer
## Preparation
1. git clone
2. download cfgs and weights for SSD512 and GLCIC
```
# SSD weights
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel  
# SSD cfgs
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/cfgs/deploy.prototxt  
# GLCIC weights
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/weights/completionnet_places2.pth
```
In terms of SSD, We use [Weiliu's SSD model](https://github.com/weiliu89/caffe/tree/ssd). So you can download SSD's cfgs and weights from [this](https://github.com/weiliu89/caffe/tree/ssd).  
GLCIC model we use is [Iizuka's model](https://github.com/satoshiiizuka/siggraph2017_inpainting).  
    - We convert the model's type from torch7 to pytorch.

## Usage
1. custom config in ganonymizer/main.py
2. ```python -m ganonymizer.main```


## Detail of the paper
Title: GANonymizer: Image Anonymization Method Integrating Object detection and Generative Adversarial Networks  
Authors: Tomoki Tanimura, Makoto Kawano, Takuro Yonezawa, Jin Nakazawa


## Reference
[1] https://github.com/weiliu89/caffe/tree/ssd  
[2] https://github.com/satoshiiizuka/siggraph2017_inpainting
