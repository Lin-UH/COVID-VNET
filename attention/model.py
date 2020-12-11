import datetime
from glob import glob
import math
from itertools import cycle

import numpy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pydicom
from tf_keras_vis.utils import normalize
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_core.python.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
import sys
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tqdm import tqdm
import random
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential,Model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import classification_report,confusion_matrix
import time
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
# config = tensorflow.compat.v1.ConfigProto(allow_soft_placement=True)
#
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# tensorflow.compat.v1.keras.backend.set_session(tensorflow.compat.v1.Session(config=config))
import prepare_dataset
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
def make_fusion_model(name,plotmodel=False,inumber="",threshold = ""):
    if name=='VGG16':
        from tensorflow.keras.applications.vgg16 import VGG16
        base_model1 = VGG16(include_top=False, input_shape=(224, 224, 3,), layers=tf.keras.layers)
        tempmodel1 = prepare_dataset.make_model("VGG16", input_shape=(224, 224, 3,), num_classes=4)
        tempmodel1.load_weights('D:/MLfinal_project/savedmodel/Segmentation/VGG16/save_at_11.h5')
        for i in range(0,19):
            base_model1.layers[i].set_weights=[tempmodel1.layers[i].get_weights]
        base_model1.trainable = False

        base_model2 = VGG16(include_top=False, input_shape=(224, 224, 3,), layers=tf.keras.layers)
        tempmodel2 = prepare_dataset.make_model("VGG16", input_shape=(224, 224, 3,), num_classes=4)
        tempmodel2.load_weights('D:/MLfinal_project/attention/savedmodel/localbranch/VGG16/07/save_at_13.h5')
        for i in range(0, 19):
            base_model2.layers[i].set_weights = [tempmodel2.layers[i].get_weights]
        for layer in base_model2.layers:
            layer._name = layer.name + str("_2")
        base_model2.trainable = False
    elif name=="ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        base_model1 = ResNet50(include_top=False, input_shape=(224, 224, 3,), layers=tf.keras.layers,pooling="avg")
        tempmodel1 = prepare_dataset.make_model("ResNet50", input_shape=(224, 224, 3,), num_classes=4)
        tempmodel1.load_weights('D:/MLfinal_project/savedmodel/Segmentation/ResNet50/save_at_9.h5')
        print(len(tempmodel1.layers))
        for i in range(0, 176):
            base_model1.layers[i].set_weights = [tempmodel1.layers[i].get_weights]
        base_model1.trainable = False
        base_model2 = ResNet50(include_top=False, input_shape=(224, 224, 3,), layers=tf.keras.layers,pooling="avg")
        tempmodel2 = prepare_dataset.make_model("ResNet50", input_shape=(224, 224, 3,), num_classes=4)
        tempmodel2.load_weights(
            'D:/MLfinal_project/attention/savedmodel/localbranch/ResNet50/'+threshold +'/save_at_'+inumber+'.h5')
        for i in range(0, 176):
            base_model2.layers[i].set_weights = [tempmodel2.layers[i].get_weights]
        for layer in base_model2.layers:
            layer._name = layer.name + str("_2")
        base_model2.trainable = False
    #
    # elif name=="VGG19":
    #     from tensorflow.keras.applications.vgg19 import VGG19
    #     base_model1=base_model2 = VGG19(include_top=False, input_shape=input_shape, weights='imagenet')
    # elif name=="ResNet50":
    #     from tensorflow.keras.applications.resnet50 import ResNet50
    #     from tensorflow.keras.applications.vgg19 import VGG19
    #     base_model1=ResNet50(include_top=False, input_shape=input_shape)
    #     tempmodel1=make_model("ResNet50",input_shape=(224, 224, 3,), num_classes=4)
    #     tempmodel1.load_weights('D:/MLfinal_project/savedmodel/Segmentation/ResNet50/save_at_9.h5')
    #     for i in range(1,107):
    #         base_model1.layers[i].set_weights=[tempmodel1.layers[i].get_weights]
    #
    #     # base_model1.load_weights('D:/MLfinal_project/savedmodel/ResNet50/save_at_9.h5')
    #     base_model1.trainable = False
    #     tempmodel2 = make_model("VGG19", input_shape=(224, 224, 3,), num_classes=4)
    #     tempmodel2.load_weights('D:/MLfinal_project/attention/savedmodel/VGG19/save_at_24.h5')
    #     base_model2 = VGG19(include_top=False, input_shape=input_shape)
    #     # for layer in model.layers:
    #     #     layer._name = layer.name + str("_2")
    #     for i in range(1,17):
    #         base_model2.layers[i].set_weights=[tempmodel2.layers[i].get_weights]
    #     # base_model2.load_weights('D:/MLfinal_project/attention/savedmodel/VGG19/save_at_24.h5')
    #     base_model2.trainable = False
    #     # print(base_model2.summary())
    # elif name=="InceptionV3":
    #     from tensorflow.keras.applications.inception_v3 import InceptionV3
    #     base_model1 = base_model2=InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet')
    # elif name=="MobileNet":
    #     from tensorflow.keras.applications.mobilenet import MobileNet
    #     base_model1=base_model2 = MobileNet(include_top=False, input_shape=input_shape, weights='imagenet')
    # inp1 = Input(input_shape)
    # inp2 = Input(input_shape)
    # base_model1.input=Input((h, w, c))
    # base_model2.input=Input((h, w, c))

    model1=base_model1.output
    model2 = base_model2.output
    # for layer in base_model2.layers:
    #     layer.name = layer.name + str("_2")
    # base_model2.
    # model2 = base_model2.layers[-1].output
    # preds=model1
    # xmax = 0
    # xmin = 6
    # ymax = 0
    # ymin = 6
    #
    # preds = preds-tf.keras.backend.min(preds)/(tf.keras.backend.max(preds)-tf.keras.backend.min(preds))
    #     # normalize(preds.numpy())
    # vertex = np.zeros((preds.shape[0], 4)).astype(int)
    # for each in range(preds.shape[0]):
    #     for x in range(preds.shape[1]):
    #         for y in range(preds.shape[2]):
    #             for index in range(preds.shape[3]):
    #                 if preds[each, x, y, index] >= threshold:
    #
    #                     if x <= xmin and y <= ymin:
    #                         xmin = x
    #                         ymin = y
    #                     if x >= xmax and y >= ymax:
    #                         xmax = x
    #                         ymax = y
    #                     continue
    #     vertex[each] = [xmin, ymin, xmax, ymax]
    # # print(vertex)
    # newimages=np.zeros((preds.shape[0],224,224,3))
    # for each in range(newimages.shape[0]):
    #     newimages[each:] = cv2.resize(base_model1.input[each][vertex[each][0] * 37:vertex[each][2] * 37, vertex[each][1] * 37:vertex[each][3] * 37],
    #                      (224, 224))
    # # newimages=np.expand_dims(newimages,axis=-1)
    model = keras.layers.concatenate([model1,model2])
    # model = layers.Dense(512, activation='relu', name="Dense1")(model)
    # model = layers.Dense(256, activation='relu', name="Dense2")(model)
    model = layers.Dense(4, activation='softmax', name="Dense3")(model)
    headmodel = Model(inputs=[base_model1.input,base_model2.input], outputs=model)
    base_model2.trainable = False
    base_model1.trainable = False
    if plotmodel==True:
        from tensorflow.keras.utils import plot_model
        plot_model(model,to_file='model_auth.png',show_shapes=True)
    print(headmodel.summary())
    return headmodel
# def make_model(name,input_shape,num_classes):
#     if name=="VGG19":
#         from tensorflow.keras.applications.vgg19 import VGG19
#         base_model = VGG19(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="ResNet50":
#         from tensorflow.keras.applications.resnet50 import ResNet50
#         base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="InceptionV3":
#         from tensorflow.keras.applications.inception_v3 import InceptionV3
#         base_model =InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name=="MobileNet":
#         from tensorflow.keras.applications.mobilenet import MobileNet
#         base_model = MobileNet(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers)
#     elif name == "VGG16":
#         from tensorflow.keras.applications.vgg16 import VGG16
#         base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet', layers=tf.keras.layers)
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
#     # print(headmodel.summary())
#     return headmodel
# =make_fusion_model(name="ResNet50",input_shape=(224, 224, 3,),num_classes=4, threshold = 0.15,plotmodel=False),
def train_fusion_model(model,name,threshold):
    if not os.path.exists('./savedmodel/globalbranch/'+name+'/'+threshold):
        os.makedirs('./savedmodel/globalbranch/'+name+'/'+threshold)
    train_datagen = ImageDataGenerator(
        # rotation_range=10,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        rescale=1. / 255
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    class ImageWithNames(DirectoryIterator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.filenames_np = np.array(self.filepaths)
            # self.class_mode = None  # so that we only get the images back

        def _get_batches_of_transformed_samples(self, index_array):
            return (super()._get_batches_of_transformed_samples(index_array),
                    self.filenames_np[index_array])
    def generate_generator_multiple(generator, dir1, dir2):
        # random.seed(1)
        train_generator =ImageWithNames(dir1, generator,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical',
            seed = 7)
        #     generator.flow_from_directory(
        #     dir1,
        #     target_size=(224, 224),
        #     batch_size=8,
        #     class_mode='categorical',
        #     seed = 7
        # )
        train1_generator =ImageWithNames(dir2, generator,
            target_size=(224, 224),
            batch_size=8,
            class_mode='categorical',
            seed = 7)
        #     generator.flow_from_directory(
        #     dir2,
        #     target_size=(224, 224),
        #     batch_size=8,
        #     class_mode='categorical',
        #     seed=7
        # )
        while True:
            X1i,namelist1=train_generator.next()
            # print(namelist1)
            X2i,namelist2=train1_generator.next()
            # print(X1i[1],X2i[1])
            # print(namelist2)
            # X1i=train_generator._get_batches_of_transformed_samples(np.array([0,5,6,11,33,45,77,32]))
            # X2i=train1_generator._get_batches_of_transformed_samples(np.array([0,5,6,11,33,45,77,32]))
            # X1i = train_generator.next()
            # X2i = train1_generator.next()
            yield [X1i[0], X2i[0]], X1i[1]  # Yield both images and their mutual label

    train_generator = generate_generator_multiple(generator=train_datagen,
                                                 dir1= 'D:/dataset/Segmentation/train',
                                                 dir2= 'D:/dataset/Crop/'+name+'/'+threshold+'/train')

    validation_generator = generate_generator_multiple(generator=test_datagen,
                                                       dir1='D:/dataset/Segmentation/val',
                                                       dir2='D:/dataset/Crop/'+name+'/'+threshold+'/val')
    earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=6, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('./savedmodel/globalbranch/'+name+'/'+threshold+'/save_at_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model on dataset
    history=model.fit_generator(
                        #     generator=training_generator,
                        # validation_data=validation_generator,
                                generator=train_generator,
                                validation_data=validation_generator,
                                steps_per_epoch=400,
                        epochs=80,
                        workers=1,
                        max_queue_size=100,validation_steps=132,
                        use_multiprocessing=False,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
               )
    training_loss=history.history["loss"]
    train_acc=history.history["accuracy"]
    test_loss=history.history["val_loss"]
    test_acc=history.history["val_accuracy"]
    epoch_count=range(1,len(training_loss)+1)
    plt.plot(epoch_count,training_loss,'r--')
    plt.plot(epoch_count,test_loss,'b--')
    plt.legend(["Training_loss","Test_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.savefig('./savedmodel/globalbranch/'+name+'/'+threshold+'/loss.jpg')
    plt.show()
    plt.plot(epoch_count,train_acc,'r--')
    plt.plot(epoch_count,test_acc,'b--')
    plt.legend(["train_acc","test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.savefig('./savedmodel/globalbranch/'+name+'/'+threshold+'/acc.jpg')
    plt.show()
def trainmodel(model = prepare_dataset.make_model("VGG19",input_shape=(224, 224, 3,), num_classes=4),savepath="VGG19",trainsetpath='D:/dataset/Crop/train/',valsetpath='D:/dataset/Crop/val/'):
    if not os.path.exists('./savedmodel/localbranch/'+savepath):
        os.makedirs('./savedmodel/localbranch/'+savepath)
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        trainsetpath,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        valsetpath,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')
    earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=6, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('./savedmodel/localbranch/'+savepath+'/save_at_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model on dataset
    history=model.fit_generator(
                        #     generator=training_generator,
                        # validation_data=validation_generator,
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=400,
        epochs=80,
        workers=8,
        max_queue_size=100, validation_steps=132,

        use_multiprocessing=False,
        callbacks=[earlyStopping, mcp_save, reduce_lr_loss]
               )
    training_loss=history.history["loss"]
    train_acc=history.history["accuracy"]
    test_loss=history.history["val_loss"]
    test_acc=history.history["val_accuracy"]
    epoch_count=range(1,len(training_loss)+1)
    plt.plot(epoch_count,training_loss,'r--')
    plt.plot(epoch_count,test_loss,'b--')
    plt.legend(["Training_loss","Test_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.savefig('./savedmodel/localbranch/'+savepath+'/loss.jpg')
    plt.show()
    plt.plot(epoch_count,train_acc,'r--')
    plt.plot(epoch_count,test_acc,'b--')
    plt.legend(["train_acc","test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.savefig('./savedmodel/localbranch/'+savepath+'/acc.jpg')
    plt.show()


def saveimage(root,f,middle_layer_model,threshold,savepath):
    images = np.asarray(np.array(cv2.resize(cv2.imread('' + root + '' + f + ''), (224, 224)) / 255))
    images = np.expand_dims(np.array(images), 0)
    preds = middle_layer_model.predict(images)

    xmax = 0
    xmin = preds.shape[1] - 1
    ymax = 0
    ymin = preds.shape[2] - 1
    preds = normalize(preds)
    vertex = np.zeros((preds.shape[0], 4)).astype(int)
    for each in range(preds.shape[0]):
        for x in range(preds.shape[1]):
            for y in range(preds.shape[2]):
                for index in range(preds.shape[3]):
                    if preds[each, x, y, index] >= threshold:
                        if x <= xmin and y <= ymin:
                            xmin = x
                            ymin = y
                        if x >= xmax and y >= ymax:
                            xmax = x
                            ymax = y
                        continue
        vertex[each] = [xmin, ymin, xmax, ymax]
    # print(vertex)
    makedirs(savepath)
    for i in range(preds.shape[0]):
        try:
            cv2.imwrite(savepath + '' + f + '.png',
                        cv2.resize(images[i][vertex[i][0] * 37:vertex[i][2] * 37, vertex[i][1] * 37:vertex[i][3] * 37],
                                   (224, 224)) * 255)
        except:
            cv2.imwrite(savepath + '' + f + '.png', cv2.resize(
                images[i][1 * 37:6 * 37, 1 * 37:6 * 37],
                (224, 224)) * 255)
import prepare_dataset
def generate_cropdataset(threshold=0.1,split="val"):
    name="ResNet50"
    model =prepare_dataset.make_model(name, input_shape=(224, 224, 3,), num_classes=4)
    # model=Model(inputs=model.inputs,outputs=model)
    Weights = 'D:/MLfinal_project/savedmodel/Segmentation/'+name+'/save_at_9.h5'
    model.load_weights(Weights)
    print(model.summary())
    data_dir = ['D:/dataset/Segmentation/'+split+'/COVID/','D:/dataset/Segmentation/'+split+'/NORMAL/','D:/dataset/Segmentation/'+split+'/PNEUMONIA/','D:/dataset/Segmentation/'+split+'/VIRUS/']
    middle_layer_model = Model(inputs=model.input, outputs=model.layers[-4].output)
    for eachpath in data_dir:
        savepath = eachpath.replace("Segmentation", 'Crop/'+name+'/'+str(threshold)+'')
        for root, dirs, files in os.walk(eachpath):
            for f in sorted(files):
                images=np.asarray(np.array(cv2.resize(cv2.imread(''+root+'' + f + ''), (224, 224)) / 255))
                images=np.expand_dims(np.array(images),0)
                preds=middle_layer_model.predict(images)
                xmax=0
                xmin=preds.shape[1]-1
                ymax=0
                ymin=preds.shape[2]-1
                preds = np.mean(preds, axis=-1)
                preds = normalize(preds)
                # for each in range(preds.shape[0]):
                #     for x in range(preds.shape[1]):
                #         for y in range(preds.shape[2]):
                #                 print(np.max(preds[each,x,y]))
                vertex=np.zeros((preds.shape[0],4)).astype(int)
                for each in range(preds.shape[0]):
                    for x in range(preds.shape[1]):
                        for y in range(preds.shape[2]):
                                if preds[each,x,y]>=threshold:
                                    if x<=xmin and y<=ymin:
                                        xmin=x
                                        ymin=y
                                    if x >= xmax and y >= ymax:
                                        # print(x,y)
                                        xmax = x
                                        ymax = y
                                    continue
                    vertex[each]=[xmin,ymin,xmax,ymax]
                # print(vertex)
                makedirs(savepath)
                for i in range(preds.shape[0]):
                    try:
                        cv2.imwrite(savepath+''+f+'.png',cv2.resize(images[i][vertex[i][0]*37:vertex[i][2]*37, vertex[i][1]*37:vertex[i][3]*37],(224,224))*255)
                    except:
                        cv2.imwrite(savepath + '' + f + '.png', cv2.resize(
                            images[i][1 * 37:6 * 37,  1* 37:6 * 37],
                            (224, 224)) * 255)
def test_fusion_model(model,Weights,threshold):
    model.load_weights(Weights)
    root_for_covid = 'D:/dataset/Covid/'
    rootpath = 'D:/dataset/Other/'
    generate_dataset = (
        lambda split: [root_for_covid + split, rootpath + split + 'NORMAL/',rootpath + split + 'PNEUMONIA/', rootpath + split + 'VIRUS/'])
    # testset path
    testset_filelist = generate_dataset('test/')
    X = np.empty((2110, 224,224, 3))
    X1 = np.empty((2110, 224, 224, 3))
    y = np.empty((2110), dtype=int)
    partition_test,labels_test=prepare_dataset.generate_partitionandlabel(testset_filelist,3000)
    for ID,i in zip(partition_test,range(len(partition_test))):
        y[i]=labels_test[ID]
        if labels_test[ID] == 0:
            X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/COVID/' + ID + '.png'), (224, 224))/255
            X1[i,] = cv2.resize(cv2.imread('D:/dataset/Crop/ResNet50/'+threshold+'/test/COVID/' + ID + '.png.png'), (224, 224)) / 255
        else:
            if labels_test[ID] == 1:
                X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/NORMAL/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/test/NORMAL/' + ID + '.png.png')) / 255
            elif labels_test[ID] == 2:
                X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/PNEUMONIA/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/test/PNEUMONIA/' + ID + '.png'
                                                                                                               '.png')) / 255
            else:
                X[i,] = cv2.resize(cv2.imread('D:/dataset/Segmentation/test/VIRUS/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/test/VIRUS/' + ID + '.png.png') )/ 255
    Y=keras.utils.to_categorical(y, num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    Score=model.evaluate([X,X1],Y,verbose=0)
    print("Test loss",Score[0])
    print("Test accuracy",Score[1])
    predictions = model.predict([X,X1])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), predictions.ravel())
    print(Y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(4):
    #     mean_tpr += interp1d(fpr[i], tpr[i],all_fpr)
    #
    # # Finally average it and compute AUC
    # mean_tpr /= 4
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"black"])
    for i, color in zip(range(4), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(''+Weights+'_acc.jpg')
    plt.show()



    print(predictions[0])
    predictions=np.argmax(predictions , axis=1)
    print(predictions[:15])
    predictions = predictions.reshape(1, -1)[0]
    print(predictions[:15])
    print(y[:15])
    print(classification_report(y, predictions, target_names=['Covid (Class 0)', 'PNEUMONIA (Class 1)','VIRUS (Class 2)','NORMAL (Class 3)']))
    print(confusion_matrix(y_true=y, y_pred=predictions, labels=[0, 1,2,3]))
    path_file_name = ''+Weights+'_result''.txt'
    resultdict=classification_report(y, predictions,
                          target_names=['Covid (Class 0)', 'PNEUMONIA (Class 1)', 'VIRUS (Class 2)',
                                        'NORMAL (Class 3)'])
    f=open(path_file_name, "w")
    f.write(resultdict)
    f.close()
def val_fusion_model(model,Weights,threshold):
    model.load_weights(Weights)
    root_for_covid = 'D:/dataset/Covid/'
    rootpath = 'D:/dataset/Other/'
    generate_dataset = (
        lambda split: [root_for_covid + split, rootpath + split + 'NORMAL/',rootpath + split + 'PNEUMONIA/', rootpath + split + 'VIRUS/'])
    # testset path
    testset_filelist = generate_dataset('val/')
    X = np.empty((2110, 224,224, 3))
    X1 = np.empty((2110, 224, 224, 3))
    y = np.empty((2110), dtype=int)
    partition_test,labels_test=prepare_dataset.generate_partitionandlabel(testset_filelist,3000)
    for ID,i in zip(partition_test,range(len(partition_test))):
        y[i]=labels_test[ID]
        if labels_test[ID] == 0:
            X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/val/COVID/' + ID + '.png'), (224, 224))/255
            X1[i,] = cv2.resize(cv2.imread('D:/dataset/Crop/ResNet50/'+threshold+'/val/COVID/' + ID + '.png.png'), (224, 224)) / 255
        else:
            if labels_test[ID] == 1:
                X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/val/NORMAL/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/val/NORMAL/' + ID + '.png.png')) / 255
            elif labels_test[ID] == 2:
                X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/val/PNEUMONIA/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/val/PNEUMONIA/' + ID + '.png'
                                                                                                               '.png')) / 255
            else:
                X[i,] = cv2.resize(cv2.imread('D:/dataset/Segmentation/val/VIRUS/' + ID + '.png'), (224, 224))/255
                X1[i,] = np.array(cv2.imread('D:/dataset/Crop/ResNet50/'+str(threshold)+'/val/VIRUS/' + ID + '.png.png') )/ 255
    Y=keras.utils.to_categorical(y, num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    Score=model.evaluate([X,X1],Y,verbose=0)
    print("Test loss",Score[0])
    print("Test accuracy",Score[1])
    f=open("D:/MLfinal_project/attention/savedmodel/globalbranch/ResNet50/val.txt", "a+")
    f.write(threshold+"  "+str(Score[1]))
    f.close()
if __name__ == '__main__':
    # for split in [ "train","val","test"]:
    #     for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #         generate_cropdataset(threshold=threshold, split=split)

    # for threshold in ["0.7","0.8","0.9"]:
    #     trainmodel(model=prepare_dataset.make_model("ResNet50", input_shape=(224, 224, 3,), num_classes=4),
    #                savepath='ResNet50/' + threshold + '',
    #                trainsetpath='D:/dataset/Crop/ResNet50/' + threshold + '/train/',
    #                valsetpath='D:/dataset/Crop/ResNet50/' + threshold + '/val/')
    # make_fusion_model("ResNet50",plotmodel=False)
    # test_fusion_model(make_fusion_model("ResNet50",plotmodel=False), './savedmodel/globalbranch/ResNet50/0.9030899869919434/save_at_13.h5', "0.9030899869919434")
    # inumber={"0.1":"11","0.2":"10","0.3":"12","0.4":"9","0.5":"15","0.6":"15","0.7":"13", "0.8":"13", "0.9":"13"}
    # for threshold in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7", "0.8", "0.9"]:
    #     train_fusion_model(make_fusion_model("ResNet50",plotmodel=False,inumber=inumber[threshold],threshold=threshold),"ResNet50",threshold=threshold)
    # inumber = {
    #            "0.9": "13"}
    # for threshold in [ "0.9"]:
    #     train_fusion_model(
    #         make_fusion_model("ResNet50", plotmodel=False, inumber=inumber[threshold], threshold=threshold), "ResNet50",
    #         threshold=threshold)
    inumber = {"0.1": "11", "0.2": "10", "0.3": "12", "0.4": "9", "0.5": "15", "0.6": "15", "0.7": "13", "0.8": "13",
               "0.9": "13"}
    # test_fusion_model(make_fusion_model("ResNet50", plotmodel=False, inumber=inumber["0.8"], threshold="0.8"),
    #                   './savedmodel/globalbranch/ResNet50/0.8/save_at_6.h5', "0.8")
    glonumber = {"0.1": "12", "0.2": "13", "0.3": "13", "0.4": "13", "0.5": "13", "0.6": "7", "0.7": "13", "0.8": "6",
               "0.9": "13"}
    for threshold in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]:
        val_fusion_model(make_fusion_model("ResNet50", plotmodel=False, inumber=inumber[threshold], threshold=threshold),
                      './savedmodel/globalbranch/ResNet50/'+threshold+'/save_at_'+glonumber[threshold]+'.h5',threshold)
def showboxonimage():
    model =make_model("ResNet50", input_shape=(224, 224, 3,), num_classes=4)
    print(model.summary())
    # model=Model(inputs=model.inputs,outputs=model)
    Weights = 'D:/MLfinal_project/savedmodel/ResNet50/save_at_9.h5'
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
    threshold=0.15
    xmax=0
    xmin=6
    ymax=0
    ymin=6
    preds=normalize(preds)
    vertex=np.zeros((5,4)).astype(int)
    for each in range(preds.shape[0]):
        for x in range(preds.shape[1]):
            for y in range(preds.shape[3]):
                for index in range(preds.shape[2]):
                    if preds[each,x,y,index]>=threshold:
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
        cv2.rectangle(images[i], (vertex[i][0]*37, vertex[i][1]*37), (vertex[i][2]*37, vertex[i][3]*37), (0, 0, 255), 2)
        ax[i].imshow(images[i])
    plt.tight_layout()
    plt.show()
    # for i in range(5):
    #     tem=cv2.resize(images[i][vertex[i][0]*37:vertex[i][2]*37, vertex[i][1]*37:vertex[i][3]*37],(224,224))
    #     cv2.imshow("s",tem)
    #     cv2.waitKey(0)
    # plt.show()