import os
from matplotlib import pyplot as plt
import cv2
import pydicom
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
def turn_covid_to_Png(split):
    root_for_covid = 'D:/dataset/Covid/'+split+'/'
    if not os.path.exists('D:/dataset/JPG/'+split+'/COVID/'):
        os.makedirs('D:/dataset/JPG/'+split+'/COVID/')
    # print(root_for_covid)
    for root, dirs, files in os.walk(root_for_covid):
        for f in sorted(files):
            eachpath = os.path.join(root, f)
            ds = pydicom.read_file(eachpath)

            try:
                if ds[0x0028, 0x0004].value == "MONOCHROME1":
                    # print(ds.pixel_array.dtype)
                    h = np.invert(ds.pixel_array)
                    small = np.min(h)
                    high = np.max(h)
                    image = (h - small) / (high - small)
                else:
                    h = ds.pixel_array
                    # print(ds[0x0028, 0x0004].value, ds[0x0028, 0x1050].value, ds[0x0028, 0x1051].value)
                    h[h <= (ds[0x0028, 0x1050].value - ds[0x0028, 0x1051].value / 2)] = 0
                    h[h >= (ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)] = ds[0x0028, 0x1050].value + ds[
                        0x0028, 0x1051].value / 2
                    image = (h) / (ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)
            except:
                if ds[0x0028, 0x0004].value == "MONOCHROME1":
                    h = np.invert(ds.pixel_array)
                else:
                    h = ds.pixel_array
                small = np.min(h)
                high = np.max(h)
                image = (h - small) / (high - small)
            # try:
            #     print(ds[0x0028, 0x1050].value, ds[0x0028, 0x1051].value)
            #     ds.pixel_array[ds.pixel_array<=(ds[0x0028, 0x1050].value-ds[0x0028, 0x1051].value/2)]=0
            #     ds.pixel_array[ds.pixel_array >=(ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)] = ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2
            # except:
            #     small = np.min(ds.pixel_array)
            #     high = np.max(ds.pixel_array)
            #     newds = (ds.pixel_array - small) / (high - small)
            # else:
            #     newds = (ds.pixel_array) / (ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)
            cv2.imwrite('D:/dataset/JPG/'+split+'/COVID/'+f+'.png', image*255)
            # small = np.min(ds.pixel_array)
            # high = np.max(ds.pixel_array)
            # newds = (ds.pixel_array - small) / (high - small)
            # cv2.imwrite('D:/dataset/JPG/'+split+'/COVID/'+f+'.png',newds*255)
def png(split):
    root_for_covid = 'D:/dataset/Covid/'+split+'/'
    if not os.path.exists('D:/dataset/1/'+split+'/'):
        os.makedirs('D:/dataset/1/'+split+'/')
    # print(root_for_covid)
    for root, dirs, files in os.walk(root_for_covid):
        for f in sorted(files):
            eachpath = os.path.join(root, f)
            ds = pydicom.read_file(eachpath)
            try:
                print(ds[0x0028, 0x1050].value, ds[0x0028, 0x1051].value)
                ds.pixel_array[ds.pixel_array<=ds[0x0028, 0x1050].value-ds[0x0028, 0x1051].value/2]=0
                ds.pixel_array[ds.pixel_array >=ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2] = ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2
            except:
                small = np.min(ds.pixel_array)
                high = np.max(ds.pixel_array)
                newds = (ds.pixel_array - small) / (high - small)
            else:
                newds = (ds.pixel_array) / (ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)
            cv2.imwrite('D:/dataset/1/' + split + '/' + f + '.png', newds*255)
if __name__=='__main__':
    for each in ['val','test','train']:
        turn_covid_to_Png(each)
    # png("val")
    # for root, dirs, files in os.walk('D:/dataset/Covid/val/'):
    #     for f in sorted(files):
    #         eachpath = os.path.join(root, f)
    #         ds = pydicom.read_file(eachpath)
    #         print(ds)
    #         print(ds[0x0028, 0x0004].value)
    #         try:
    #             print(ds[0x0028, 0x1050].value,ds[0x0028, 0x1051].value)
    #         except:
    #             print("no")
            # break
