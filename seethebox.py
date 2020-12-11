# import os
#
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import pydicom
# import skimage
# from scipy import ndimage
#
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import *
# from tensorflow.keras import backend as keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras import layers
# from glob import glob
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow.keras.applications.vgg16 import VGG16
# def make_model(name,input_shape,num_classes):
#     if name=="VGG19":
#         from tensorflow.keras.applications.vgg19 import VGG19
#         base_model = VGG19(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="ResNet50":
#         from tensorflow.keras.applications.resnet50 import ResNet50
#         base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="VGG16":
#         from tensorflow.keras.applications.vgg16 import VGG16
#         base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="InceptionV3":
#         from tensorflow.keras.applications.inception_v3 import InceptionV3
#         base_model =InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="MobileNet":
#         from tensorflow.keras.applications.mobilenet import MobileNet
#         base_model = MobileNet(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     # model = Sequential()
#     # model.add(base_model)
#     # model.add(layers.Flatten())
#     # model.add(layers.Dense(256, activation='relu', name="Dense1"))
#     # model.add(layers.Dense(num_classes, activation='softmax', name="Dense2"))
#     # print(model.summary())
#     model=base_model.output
#     model=layers.Flatten()(model)
#     model=layers.Dense(256, activation='relu', name="Dense1")(model)
#     model=layers.Dense(num_classes, activation='softmax', name="Dense2")(model)
#     headmodel=Model(inputs=base_model.input,outputs=model)
#     base_model.trainable = False
#     return headmodel
# base_model1 = VGG16(include_top=False, input_shape=(224, 224, 3,), layers=tf.keras.layers)
# print(len(base_model1.layers))
# print(base_model1.layers[19].name)
# tempmodel1 = make_model("VGG16", input_shape=(224, 224, 3,), num_classes=4)
# print(len(tempmodel1.layers))
# print(tempmodel1.summary())
# import cv2
# import numpy as np
# a=cv2.resize(cv2.imread('D:/dataset/Crop/ResNet50/0.9030899869919434/test/PNEUMONIA/person441_bacteria_1907.jpeg.png.png'),(224, 224)) / 255
# print(np.array(a).shape)
import os
import random

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
from tf_keras_vis.utils import normalize
from glob import glob
from tqdm import tqdm
import tensorflow
import prepare_dataset
model =prepare_dataset.make_model("ResNet50", input_shape=(224, 224, 3,), num_classes=4)
print(model.summary())
# model=Model(inputs=model.inputs,outputs=model)
# Weights = 'D:/MLfinal_project/attention/savedmodel/globalbranch/ResNet50/0.7/save_at_13.h5'
Weights = 'D:/MLfinal_project/savedmodel/Segmentation/ResNet50/save_at_9.h5'
model.load_weights(Weights)
data_dir = 'D:/dataset/Segmentation/test/COVID/'
covid_images = data_dir + '*.*'
# covid_images=data_dir+'normal/*.*'
# covid_images=data_dir+'vpneumonia/*.*'
imagePaths = glob(covid_images)
r = random.sample(imagePaths, 5)
print(r)
image_titles = ["covid", "covid", "covid", "covid", "covid"]
# for imagePath in r:
#     # extract the class label from the filename
#     image_title = imagePath.split(os.path.sep)[-2]
#     image_titles.append(image_title)
image_titles = np.array(image_titles)
print(image_titles)
images = np.asarray([np.array(cv2.resize(cv2.imread('' + fname + ''), (224, 224)) / 255) for fname in r])
# preds = model.predict(images)
middle_layer_model=Model(inputs=model.input,
outputs=model.layers[-4].output)
preds=middle_layer_model.predict(images)
print(preds.shape)
threshold=0.3
xmax=0
xmin=6
ymax=0
ymin=6

preds=np.mean(preds,axis=-1)
preds=normalize(preds)
print(preds)
vertex=np.zeros((5,4)).astype(int)
for each in range(preds.shape[0]):
    for x in range(preds.shape[1]):
        for y in range(preds.shape[2]):
            # for index in range(preds.shape[3]):
                if preds[each,x,y]>=threshold:
                    if x<=xmin and y<=ymin:
                        xmin=x
                        ymin=y
                    if x >= xmax and y >= ymax:
                        xmax = x
                        ymax = y
                    continue
    vertex[each]=[xmin,ymin,xmax,ymax]
# print(model.layers[-4].output.shape)
# vertex.dtype="int"
print(vertex)
subplot_args = {'nrows': 1, 'ncols': 5, 'figsize': (21, 9),
                'subplot_kw': {'xticks': [], 'yticks': []}}
f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=14)
    print(images[i].shape)
    cv2.rectangle(images[i], (vertex[i][0]*37, vertex[i][1]*37), (vertex[i][2]*37, vertex[i][3]*37), (255, 0, 0), thickness=5)
    ax[i].imshow(images[i])
plt.tight_layout()
plt.savefig("./savedmodel/box.jpg")
plt.show()