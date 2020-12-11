import sys
import os,shutil
from tqdm import tqdm
def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.move(srcfile,dstfile)
        print ("move %s -> %s"%( srcfile,dstfile))
filePath_normal='F:/迅雷下载/covid-chestxray-dataset-master/covid-chestxray-dataset-master/images'
for root, dirs, files in os.walk(filePath_normal):
    for f in files:
        if "covid" in f:
            eachpath = os.path.join(root, f)
            mymovefile(eachpath,'D:/dataset/external_test/')