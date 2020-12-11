import os
from matplotlib import pyplot as plt
import cv2
import pydicom
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
def turn_covid_to_npy(split):
    root_for_covid = 'D:/dataset/Covid/'+split+'/'
    if not os.path.exists('D:/dataset/Covid_npy/'+split+'/'):
        os.makedirs('D:/dataset/Covid_npy/'+split+'/')
    # print(root_for_covid)
    for root, dirs, files in os.walk(root_for_covid):
        for f in sorted(files):
            # if f=='3935110146332427995.dcm':
            #     print("d")
            #     eachpath = os.path.join(root, f)
            #     ds = pydicom.read_file(eachpath)
            #     plt.imshow(ds.pixel_array, cmap='gray')
            #     plt.show()
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
            pix = np.stack((cv2.resize(newds, (224,224)),) * 3, axis=-1)
            np.save('D:/dataset/Covid_npy/'+split+'/'+f+'.npy',pix)
            # cv2.imwrite('D:/dataset/Covidjpg/test/'+f+'.png',ds.pixel_array*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # print(f)
def turn_other_to_npy(split):
    root_for_PNEUMONIA = 'D:/dataset/Other/'+split+'/PNEUMONIA'
    root_for_NORMAL = 'D:/dataset/Other/'+split+'/NORMAL'
    root_for_VIRUS = 'D:/dataset/Other/' + split + '/VIRUS'
    if not os.path.exists('D:/dataset/Other_npy/'+split+'/PNEUMONIA/'):
        os.makedirs('D:/dataset/Other_npy/'+split+'/PNEUMONIA/')
    if not os.path.exists('D:/dataset/Other_npy/' + split + '/NORMAL/'):
        os.makedirs('D:/dataset/Other_npy/' + split + '/NORMAL/')
    if not os.path.exists('D:/dataset/Other_npy/' + split + '/VIRUS/'):
        os.makedirs('D:/dataset/Other_npy/' + split + '/VIRUS/')
    i=0
    for eachpath in [root_for_PNEUMONIA,root_for_NORMAL,root_for_VIRUS]:
        i=i+1
        for root, dirs, files in os.walk(eachpath):
            for f in sorted(files):
                # if f=='3935110146332427995.dcm':
                #     print("d")
                #     eachpath = os.path.join(root, f)
                #     ds = pydicom.read_file(eachpath)
                #     plt.imshow(ds.pixel_array, cmap='gray')
                #     plt.show()
                eachpath = os.path.join(root, f)
                jpg = cv2.imread(eachpath, 0)
                pix = np.stack((cv2.resize(jpg, (224,224)),) * 3, axis=-1)
                if i==1:
                    np.save('D:/dataset/Other_npy/'+split+'/PNEUMONIA/'+f+'.npy',pix/255)
                elif i==2:
                    np.save('D:/dataset/Other_npy/' + split + '/NORMAL/' + f + '.npy',pix/255)
                else:
                    np.save('D:/dataset/Other_npy/' + split + '/VIRUS/' + f + '.npy', pix/255)
                # cv2.imwrite('D:/dataset/Covidjpg/test/'+f+'.png',ds.pixel_array*255, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                # print(f)
if __name__=='__main__':
    for each in ['val','test','train']:
        turn_covid_to_npy(each)
        # turn_other_to_npy(each)
