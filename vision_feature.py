import os

from cv2 import cv2
from tensorflow.keras.models import Sequential,Model
from tf_keras_vis.gradcam import GradcamPlusPlus,Gradcam
from tf_keras_vis.utils import normalize
from matplotlib import pyplot as plt, cm
from tensorflow import keras
import random
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from glob import glob
import  tensorflow as tf
# Create GradCAM++ object, Just only repalce class name to "GradcamPlusPlus"
# gradcam = Gradcam(model, model_modifier, clone=False)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
def loss(output):
    # COVID-19: class  [0]
    return (output[0][0], output[1][0], output[2][0], output[3][0], output[4][0])
    # Normal: class [1]
    # return (output[0][1], output[1][1], output[2][1], output[3][1], output[4][1])
    # Viral Pneumonia: clas[2]
    # return (output[0][2], output[1][2], output[2][2], output[3][2], output[4][2])
def model_modifier(m):
    m.layers[-1].activation = keras.activations.linear
    return m
img_no = 5
name="VGG19"
mode="Segmentation"
data_dir='D:/dataset/'+mode+'/test/NORMAL/'
import prepare_dataset

# inumber = {"0.1": "11", "0.2": "10", "0.3": "12", "0.4": "9", "0.5": "15", "0.6": "15", "0.7": "13", "0.8": "13",
#            "0.9": "13"}
# glonumber = {"0.1": "12", "0.2": "13", "0.3": "13", "0.4": "13", "0.5": "13", "0.6": "7", "0.7": "13", "0.8": "6",
#              "0.9": "13"}
# import attention.model as am
#
# for threshold in [ "0.3"]:
#     model=am.make_fusion_model("ResNet50", plotmodel=False, inumber=inumber[threshold], threshold=threshold)
#     model.load_weights('./attention/savedmodel/globalbranch/ResNet50/'+threshold+'/save_at_'+glonumber[threshold]+'.h5')
# name="AG-CNN"




model=prepare_dataset.make_model(name,input_shape=(224, 224, 3,),num_classes=4)
print(model.summary())
# model=Model(inputs=model.inputs,outputs=model)


Weights='./savedmodel/'+mode+'/'+name+'/save_at_8.h5'
model.load_weights(Weights)
savepath='./savedmodel/'+mode+'/'+name+'/'
covid_images = data_dir + '*.*'


# covid_images=data_dir+'normal/*.*'
# covid_images=data_dir+'vpneumonia/*.*'
imagePaths = glob(covid_images)
r = random.sample(imagePaths, img_no)
print(r)
image_titles = ["covid","covid","covid","covid","covid"]
# for imagePath in r:
#     # extract the class label from the filename
#     image_title = imagePath.split(os.path.sep)[-2]
#     image_titles.append(image_title)
image_titles = np.array(image_titles)
print(image_titles)
images = np.asarray([np.array(cv2.resize(cv2.imread('' + fname + ''), (224, 224))/255) for fname in r])

imagePaths = glob('D:/dataset/Crop/ResNet50/0.3/test/COVID/'+'*.*')
r = random.sample(imagePaths, img_no)



print(images.shape)
# X = preprocess_input(images)
# X = preprocess_input(images)
X=images
print(X.shape)
subplot_args = { 'nrows': 1, 'ncols':5, 'figsize': (21, 9),
                 'subplot_kw': {'xticks': [], 'yticks': []} }
f, ax = plt.subplots(**subplot_args)
# from skimage.color import rgb2gray
#
# img_gray = rgb2gray(img)
# for i, title in enumerate(image_titles):
#     ax[i].set_title(title, fontsize=14)
#     ax[i].imshow(images[i])
# plt.savefig('X-rays.png')
# plt.show()
gradcam = Gradcam(model,
                          model_modifier,
                          clone=True)
print(model.summary())
# Generate heatmap with GradCAM++
cam = gradcam(loss,
              X,
              penultimate_layer=-2, # model.layers number
             )
cam = normalize(cam)

f, ax = plt.subplots(**subplot_args)
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=14)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)

plt.tight_layout()
plt.savefig(savepath+'gradcam_normal.png')
plt.show()
# for i, title in enumerate(image_titles):
#     heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
#     ax[i].set_title(title, fontsize=14)
#     ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
# plt.tight_layout()
# plt.show()