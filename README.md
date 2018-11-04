# GANonymizer
## Preparation
1. git clone
```
git clone git@github.com:tanimutomo/ganonymizer.git
```
2. download cfgs and weights for SSD512 and GLCIC
```
# cd project root
cd ganonymizer
# SSD cfgs
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/cfgs/deploy.prototxt -P ganonymizer/src/detection/ssd/cfgs 
# SSD weights
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/ssd/weights/VGG_VOC0712Plus_SSD_512x512_iter_240000.caffemodel -P ganonymizer/src/detection/ssd/weights
# GLCIC weights
wget https://www.ht.sfc.keio.ac.jp/~tanimu/ganonymizer/glcic/weights/completionnet_places2.pth -P ganonymizer/src/inpaint/glcic/weights
```
In terms of SSD, We use [Weiliu's SSD model](https://github.com/weiliu89/caffe/tree/ssd). So you can download SSD's cfgs and weights from [this](https://github.com/weiliu89/caffe/tree/ssd).  
GLCIC model we use is [Iizuka's model](https://github.com/satoshiiizuka/siggraph2017_inpainting).  
    - We convert the model's type from torch7 to pytorch.


## Usage
1. custom config in ganonymizer/main.py
2. ```python -m ganonymizer.main```


## Reference
[1] https://github.com/weiliu89/caffe/tree/ssd  
[2] https://github.com/satoshiiizuka/siggraph2017_inpainting


## Details of this paper
Title: GANonymizer: Image Anonymization Method Integrating Object detection and Generative Adversarial Networks  
Authors: Tomoki Tanimura, Makoto Kawano, Takuro Yonezawa, Jin Nakazawa
