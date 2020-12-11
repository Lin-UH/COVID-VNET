import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
import skimage
from scipy import ndimage

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from glob import glob
from tqdm import tqdm
import tensorflow
config = tensorflow.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.3
tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config))
def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
# model = unet(input_size=(512,512,1))
# model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,
#                   metrics=[dice_coef, 'binary_accuracy'])
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
def generate():
    model = load_model('./savedmodel/unet_lung_seg.hdf5',
                       custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    for split in ["val","test","train"]:
        for path in ['D:/dataset/COVID/'+split+'/','D:/dataset/JPG/'+split+'/NORMAL/','D:/dataset/JPG/' + split + '/PNEUMONIA/','D:/dataset/JPG/' + split + '/VIRUS/']:
            makedirs(path)
            if "COVID" in path:
                for root, dirs, files in os.walk(path):
                    for f in sorted(files):
                        print(f)
                        ds = pydicom.read_file(os.path.join(root, f))
                        try:
                            if ds[0x0028, 0x0004].value == "MONOCHROME1":
                                # print(ds.pixel_array.dtype)
                                h = np.invert(ds.pixel_array)
                                small = np.min(h)
                                high = np.max(h)
                                image = (h - small) / (high - small)
                            else:
                                h=ds.pixel_array
                                print(ds[0x0028, 0x0004].value ,ds[0x0028, 0x1050].value,ds[0x0028, 0x1051].value)
                                h[h<=(ds[0x0028, 0x1050].value-ds[0x0028, 0x1051].value/2)]=0
                                h[h>=(ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)] = ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2
                                image = (h) / (ds[0x0028, 0x1050].value + ds[0x0028, 0x1051].value / 2)
                        except:
                            if ds[0x0028, 0x0004].value=="MONOCHROME1":
                                h=np.invert(ds.pixel_array)
                            else:
                                h=ds.pixel_array
                            small = np.min(h)
                            high = np.max(h)
                            image = (h - small) / (high - small)

                        # cv2.imshow("img", image)
                        # cv2.waitKey(0)
                        # image=image*255

                        # image = cv2.imread(path, 0)
                        image = cv2.resize(image, (512, 512))
                        image = np.array(image).reshape(1, 512, 512, 1).astype(np.float32)
                        preds = model.predict(image)
                        #
                        # plt.figure(figsize=(30, 10))
                        # plt.subplot(1, 3, 1)
                        # plt.imshow(np.squeeze(image))
                        # plt.xlabel("Base Image")
                        # plt.subplot(1, 3, 2)
                        # plt.imshow(np.squeeze(preds))
                        # print(np.squeeze(preds))


                        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        #
                        ret, binary = cv2.threshold(np.squeeze(preds), 0.5, 1, cv2.THRESH_BINARY)
                        img = ndimage.binary_fill_holes(binary).astype(np.uint8)
                        # img=ndimage.binary_fill_holes(np.squeeze(preds)).astype(np.uint8)
                        img[img>0]=255
                        img=img.astype(np.uint8)
                        emptyimage=np.zeros((512,512))
                        # emptyimage=np.zeros((512,512)).astype(np.uint8)
                        # ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
                        # cv2.imshow("img", img)
                        # cv2.waitKey(0)
                        corrd={}
                        for i in range(0,len(contours)):
                            corrd[cv2.contourArea(contours[i])]=i
                        sorteddict=sorted(corrd.items(), key = lambda kv:(kv[0], kv[1]),reverse=True)
                        # print(sorteddict[0][1],sorteddict[1][1])
                        try:
                            newcontours=[contours[sorteddict[0][1]],contours[sorteddict[1][1]]]
                        except:
                            newcontours = [contours[sorteddict[0][1]]]
                        cv2.drawContours(emptyimage, newcontours, -1, (1,1,1), -1)
                        # cv2.imshow("img", emptyimage)
                        # cv2.waitKey(0)





                        # plt.xlabel("Mask")
                        # plt.subplot(1, 3, 3)
                        # for i in range(0,emptyimage.shape[0]):
                        #     for j in range(emptyimage.shape[1]):
                        #         if emptyimage[i][j]!=0 and  emptyimage[i][j]!=1:
                        #             print(emptyimage[i][j])
                        final=np.squeeze(image)*emptyimage*255
                        storepath='D:/dataset/Segmentation/' + split + '/COVID/'
                        makedirs(storepath)
                        cv2.imwrite(storepath + '' + f + '.png', final)
                        # plt.imshow(final)
                        # plt.xlabel("Result")
                        # plt.show()
            else:
                for root, dirs, files in os.walk(path):
                    for f in sorted(files):
                        image = cv2.imread(os.path.join(root, f), 0)
                        image = cv2.resize(image, (512, 512))
                        image = np.array(image).reshape(1, 512, 512, 1).astype(np.float32)
                        preds = model.predict(image)
                        #
                        # plt.figure(figsize=(30, 10))
                        # plt.subplot(1, 3, 1)
                        # plt.imshow(np.squeeze(image))
                        # plt.xlabel("Base Image")
                        # plt.subplot(1, 3, 2)
                        # plt.imshow(np.squeeze(preds))
                        # print(np.squeeze(preds))

                        # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        #
                        img = ndimage.binary_fill_holes(np.squeeze(preds)).astype(np.uint8)
                        img[img > 0] = 255
                        img = img.astype(np.uint8)
                        emptyimage = np.zeros((512, 512))
                        # emptyimage=np.zeros((512,512)).astype(np.uint8)
                        # ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
                        # cv2.imshow("img", img)
                        # cv2.waitKey(0)
                        corrd = {}
                        for i in range(0, len(contours)):
                            corrd[cv2.contourArea(contours[i])] = i
                        sorteddict = sorted(corrd.items(), key=lambda kv: (kv[0], kv[1]), reverse=True)
                        # print(sorteddict[0][1],sorteddict[1][1])
                        try:
                            newcontours = [contours[sorteddict[0][1]], contours[sorteddict[1][1]]]
                        except:
                            newcontours = [contours[sorteddict[0][1]]]
                        cv2.drawContours(emptyimage, newcontours, -1, (1, 1, 1), -1)
                        # cv2.imshow("img", emptyimage)
                        # cv2.waitKey(0)

                        # plt.xlabel("Mask")
                        # plt.subplot(1, 3, 3)
                        # for i in range(0,emptyimage.shape[0]):
                        #     for j in range(emptyimage.shape[1]):
                        #         if emptyimage[i][j]!=0 and  emptyimage[i][j]!=1:
                        #             print(emptyimage[i][j])
                        final = np.squeeze(image) * emptyimage
                        storepath=path.replace("JPG","Segmentation")
                        makedirs(storepath)
                        cv2.imwrite(storepath+'' + f + '.png', final)
                        # plt.imshow(final)
                        # plt.xlabel("Result")
                        # plt.show()
if __name__ == '__main__':
    generate()