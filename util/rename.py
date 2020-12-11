# -*- coding:utf8 -*-
import os

path = 'D:/dataset/Crop/train/PNEUMONIA'
filelist = os.listdir(path)
for item in filelist:
    if item.endswith('.png'):
        src = os.path.join(os.path.abspath(path), item)
        dst = os.path.join(os.path.abspath(path), item.replace(".png","",1))
    try:
        os.rename(src, dst)
        print('rename from %s to %s' % (src, dst))
    except:
        continue
