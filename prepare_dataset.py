import datetime
import numpy
import numpy as np
import pydicom
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow_core.python.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import os
from itertools import cycle
from scipy.interpolate import interp1d
import sys
import cv2
from tqdm import tqdm
import random
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential,Model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
#
# config.gpu_options.per_process_gpu_memory_fraction = 0.1
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, IMAGE_SIZE)
def walkFile(file,label,howmany,type,size=512):
    temp=[]
    tempy = []
    if type=="DICM":
        for root, dirs, files in os.walk(file):
            # root 表示当前正在访问的文件夹路径
            # dirs 表示该文件夹下的子目录名list
            # files 表示该文件夹下的文件list
            # 遍历文件
            i=0
            for i ,f in zip(tqdm(range(howmany)),files):
                    if i<howmany:
                        eachpath=os.path.join(root, f)
                        ds = pydicom.read_file(eachpath)
                        pix = cv2.resize(ds.pixel_array,(size,size))
                        temp.append(pix)
                        # plt.imshow(pix, cmap='gray')
                        # plt.show()
                        i+=1
    elif type=="JPEG":
        for root, dirs, files in os.walk(file):
            i = 0
            for i, f in zip(tqdm(range(howmany)), files):
                if i < howmany:
                    eachpath = os.path.join(root, f)
                    jpg=cv2.imread(eachpath,0)
                    pix = cv2.resize(jpg, (size, size))
                    temp.append(pix)
                    # plt.imshow(pix, cmap='gray')
                    # plt.show()
                    i += 1
    for j in range(0, howmany):
        tempy.append(label)
    return temp,tempy
    # ai32 = np.array(temp, dtype=np.uint8)
    # print("size of 0 int32 number: %f" % sys.getsizeof(ai32))
        #
        # # 遍历所有的文件夹
        # for d in dirs:
        #     print(os.path.join(root, d))
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
# define VGG16 structure
def make_VGG16_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x = layers.Conv2D(input_shape=input_shape,kernel_size=(3,3),filters=64,strides=1,padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=64, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x=layers.Dropout(0.1)(x)
    x=layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=128, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=128, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=256, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=256, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=256, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x )
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(input_shape=input_shape, activation='relu',kernel_size=(3, 3), filters=512, strides=1, padding='same',name='block5_conv3')(x)
    # x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2,name="block5_pool")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(64)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return  keras.Model(inputs, outputs)
def pretrained_model1():
    from tensorflow.keras.applications.vgg16 import VGG16
    pretrained_model1 = VGG16(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
        , layers=tf.keras.layers
    )
    for layer in pretrained_model1.layers[:12]:
        layer.trainable = False
    for layer in pretrained_model1.layers[13:]:
        layer.trainable = True

    model1 = Sequential()
    # first (and only) set of FC => RELU layers
    model1.add(layers.AveragePooling2D((2, 2), name='avg_pool'))
    model1.add(layers.Flatten())

    model1.add(layers.Dense(64, activation='relu'))
    model1.add(layers.Dropout(0.3))

    model1.add(layers.Dense(4, activation='softmax'))

    preinput1 = pretrained_model1.input
    preoutput1 = pretrained_model1.output
    output1 = model1(preoutput1)
    model1 = Model(preinput1, output1)

    model1.summary()

    return model1

def pretrained_model2():
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
    pretrained_model2 = InceptionResNetV2(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
        , layers=tf.keras.layers
    )
    for layer in pretrained_model2.layers[:-280]:
        layer.trainable = False
    model2 = Sequential()

    model2.add(layers.AveragePooling2D((2, 2), name='avg_pool'))
    model2.add(layers.Flatten())
    model2.add(layers.Dense(256, activation='relu'))
    model2.add(layers.Dropout(0.5))

    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dropout(0.3))

    model2.add(layers.Dense(4, activation='softmax'))

    preinput2 = pretrained_model2.input
    preoutput2 = pretrained_model2.output
    output2 = model2(preoutput2)
    model2 = Model(preinput2, output2)

    model2.summary()
    return model2

def pretrained_model3():
    from tensorflow.keras.applications.densenet import DenseNet121
    pretrained_model3 = DenseNet121(
        include_top=False,
        input_shape=(224, 224, 3),
        weights='imagenet'
        , layers=tf.keras.layers
    )
    for layer in pretrained_model3.layers[:-200]:
        layer.trainable = False

    model3 = Sequential()
    # first (and only) set of FC => RELU layers
    model3.add(layers.AveragePooling2D((2, 2), name='avg_pool'))
    model3.add(layers.Flatten())

    model3.add(layers.Dense(64, activation='relu'))
    model3.add(layers.Dropout(0.3))

    model3.add(layers.Dense(4, activation='softmax'))

    preinput3= pretrained_model3.input
    preoutput3= pretrained_model3.output
    output3 = model3(preoutput3)
    model3 = Model(preinput3, output3)

    model3.summary()
    return model3
# load some weight from VGG16 on imagenet
def load_VGG16_partofweights():
    from tensorflow.keras.applications.vgg16 import VGG16
    model = make_VGG16_model(input_shape=(224, 224, 3,), num_classes=4)
    base_model = VGG16(weights='imagenet', include_top=True)
    print(base_model.summary())
    print(model.summary())
    # Select the layers for which you want to set weight.
    # base_model.layers[1].get_weights()
    # base_model.get_layer(layer_name).get_weights()
    w, b = base_model.layers[1].get_weights()
    model.layers[1].set_weights = [w, b]
    w, b = base_model.layers[2].get_weights()
    model.layers[3].set_weights = [w, b]
    w, b = base_model.layers[4].get_weights()
    model.layers[8].set_weights = [w, b]
    w, b = base_model.layers[5].get_weights()
    model.layers[10].set_weights = [w, b]
    w, b = base_model.layers[7].get_weights()
    model.layers[15].set_weights = [w, b]
    w, b = base_model.layers[8].get_weights()
    model.layers[17].set_weights = [w, b]
    w, b = base_model.layers[9].get_weights()
    model.layers[19].set_weights = [w, b]
    # w,b = base_model.layers[11].get_weights()
    # model.layers[24].set_weights = [w,b]
    #
    #
    # w,b = base_model.layers[12].get_weights()
    # model.layers[26].set_weights = [w,b]
    #
    # w,b = base_model.layers[13].get_weights()
    # model.layers[28].set_weights = [w,b]
    point = -17
    # for layers in model.layers[:point]:
    #     layers.trainable=False
    # callbacks = [
    #         keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    #     ]
    return model
def make_model(name,input_shape,num_classes):
    if name=="VGG19":
        from tensorflow.keras.applications.vgg19 import VGG19
        base_model = VGG19(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers,pooling="max")
    elif name=="ResNet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers,pooling="avg")
    elif name=="VGG16":
        from tensorflow.keras.applications.vgg16 import VGG16
        base_model = VGG16(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers,pooling="max")
    elif name=="InceptionV3":
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        base_model =InceptionV3(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers,pooling="max")
    elif name=="MobileNet":
        from tensorflow.keras.applications.mobilenet import MobileNet
        base_model = MobileNet(include_top=False, input_shape=input_shape, weights='imagenet',layers=tf.keras.layers,pooling="max")
    elif name == "VGG161":
        return pretrained_model1()
    elif name=="InceptionResNetV2":
        return pretrained_model2()
    elif name=="DenseNet121":
        return pretrained_model3()
    # model = Sequential()
    # model.add(base_model)
    # model.add(layers.Flatten())
    # model.add(layers.Dense(256, activation='relu', name="Dense1"))
    # model.add(layers.Dense(num_classes, activation='softmax', name="Dense2"))
    # print(model.summary())
    model=base_model.output
    # model=layers.Flatten()(model)
    model=layers.Dense(256, activation='relu', name="Dense1")(model)
    model=layers.Dense(num_classes, activation='softmax', name="Dense2")(model)
    headmodel=Model(inputs=base_model.input,outputs=model)
    # print(headmodel.summary())
    # base_model.trainable = False
    return headmodel
# walk through file, generate two lists, temp includes filename  =[xxxx.jpg,xxxx.dic,...], tempy includes labels =[1,2,3,0,1...]
def generate_partitionandlabel(file,howmany):
    temp=[]
    tempy = {}
    for labelindex,eachpath in zip(range(len(file)),file):
        for root, dirs, files in os.walk(eachpath):
            for i, f in zip(tqdm(range(howmany)), files):
                if i < howmany:
                    temp.append(f)
                    tempy[f]=labelindex
    random.shuffle(temp)
    return temp,tempy
def generate_nyarray(file,howmany):
    temp=[]
    tempy = []
    for labelindex,eachpath in zip(range(len(file)),file):
        for root, dirs, files in os.walk(eachpath):
            for i, f in zip(tqdm(range(howmany)), files):
                    g = np.load('' + root + '' + f + '')
                    # g.dtype=np.float32
                    temp.append(g)
                    tempy.append(keras.utils.to_categorical(labelindex, num_classes=4))
    xtrain=np.array(temp)
    ytrain=np.array(tempy)
    print(xtrain.shape)
    return xtrain,ytrain
# design a DataGenerator used for save memory
class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=32, dim=(512, 512), n_channels=3,
                 n_classes=4, shuffle=True, filelist=[],split="train",mode="npy"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.filelist=filelist
        self.split=split
        self.mode=mode
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store class
            y[i] = self.labels[ID]
            # Store sample
            if self.labels[ID] == 0:
                X[i,]=np.load('D:/dataset/Covid_npy/'+self.split+'/'+ID+'.npy')
            else:
                if self.labels[ID] == 1:
                    X[i,] = np.load('D:/dataset/Other_npy/' + self.split + '/PNEUMONIA/' + ID + '.npy')
                elif self.labels[ID] == 2:
                    X[i,] = np.load('D:/dataset/Other_npy/' + self.split + '/VIRUS/' + ID + '.npy')
                else:
                    X[i,] = np.load('D:/dataset/Other_npy/' + self.split + '/NORMAL/' + ID + '.npy')
            # if self.labels[ID]==0:
            #         start_t = datetime.datetime.now()
            #         # for root, dirs, files in os.walk(self.filelist[0]):
            #         #     for f in files:
            #         #         if f==ID:
            #         eachpath = os.path.join(self.filelist[0], ID)
            #         ds = pydicom.read_file(eachpath)
            #         # pix = cv2.cvtColor(cv2.resize(ds.pixel_array, self.dim), cv2.COLOR_GRAY2BGR)
            #         pix = np.stack((cv2.resize(ds.pixel_array, self.dim),) * 3, axis=-1)
            #         X[i,] =pix/255
            #         end_t = datetime.datetime.now()
            #         print("covid")
            #         print(end_t - start_t)
            # else:
            #     # for root, dirs, files in os.walk(self.filelist[self.labels[ID]]):
            #     #     for f in files:
            #     #         if f == ID:
            #                 start_t = datetime.datetime.now()
            #                 eachpath = os.path.join(self.filelist[self.labels[ID]], ID)
            #                 jpg = cv2.imread(eachpath, 0)
            #                 # pix = cv2.cvtColor(cv2.resize(jpg, self.dim), cv2.COLOR_GRAY2BGR)
            #                 pix = np.stack((cv2.resize(jpg, self.dim),) * 3, axis=-1)
            #                 X[i,] = pix/255
            #                 end_t = datetime.datetime.now()
            #                 print("other")
            #                 print(end_t - start_t)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# trainmodel, need a parameter model
def trainmodelwithnpy(model,savepath):
    # root_for_covid='F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/'
    # rootpath='F:/Edge_download/data/archive/chest_xray/chest_xray/'
    if not os.path.exists('./savedmodel/'+savepath):
        os.makedirs('./savedmodel/'+savepath)
    root_for_covid = 'D:/dataset/Covid/'
    rootpath = 'D:/dataset/Other/'
    generate_dataset=(lambda split:[root_for_covid+split,rootpath+split+'PNEUMONIA/',rootpath+split+'VIRUS/',rootpath+split+'NORMAL/'])
    # trainset path
    trainset_filelist=generate_dataset('train/')
    # valset path
    valset_filelist=generate_dataset('val/')
    # testset path
    testset_filelist = generate_dataset('test/')

    partition_train,labels_train=generate_partitionandlabel(trainset_filelist,3000)
    training_generator = DataGenerator(partition_train, labels_train, batch_size=16, dim=(224, 224), n_channels=3,
                     n_classes=4, shuffle=True, filelist=trainset_filelist,split="train")
    partition_val,labels_val=generate_partitionandlabel(valset_filelist,3000)
    validation_generator = DataGenerator(partition_val, labels_val, batch_size=16, dim=(224, 224), n_channels=3,
                     n_classes=4, shuffle=True, filelist=valset_filelist,split="val")
    earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=15, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('./savedmodel/'+savepath+'/save_at_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model on dataset
    history=model.fit_generator(
                            generator=training_generator,
                        validation_data=validation_generator,
                                steps_per_epoch=161,
                        epochs=50,
                        workers=8,
                        max_queue_size=100,validation_steps=61,

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
    plt.show()
    # X,y=validation_generator.__getitem__(1)

    plt.plot(epoch_count,train_acc,'r--')
    plt.plot(epoch_count,test_acc,'b--')
    plt.legend(["train_acc","test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.show()

    # model.save('./newmodel.h5')
    # trainx=trainx_covid+trainx_normal+trainx_pneumonia+trainx_virus
    # trainy=trainy_covid+trainy_normal+trainy_pneumonia+trainy_virus
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(trainx, trainy, test_size = 0.3, random_state=1)
    # x_train4D = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in X_train]
    # x_train4D = np.concatenate([arr[np.newaxis] for arr in x_train4D]).astype('float32')
    #
    # x_test4D = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in X_test]
    # x_test4D = np.concatenate([arr[np.newaxis] for arr in x_test4D]).astype('float32')
    #
    # x_test4D_normalize=x_test4D/255
    # x_train4D_normalize=x_train4D/255
    #
    # """one_hot encoding"""
    # y_trainOnehot=to_categorical(Y_train)
    # y_testOnehot=to_categorical(Y_test)
    # VGG16 = make_VGG16_model(input_shape=(512,512,3,), num_classes=4)
    # keras.utils.plot_model(VGG16, show_shapes=True)
    # for step, (x, y) in enumerate(dataset):
    #     with tf.GradientTape() as tape:
    #
    #         # Forward pass.
    #         logits = mlp(x)
    #
    #         # External loss value for this batch.
    #         loss = loss_fn(y, logits)
    #         # Add the losses created during the forward pass.
    #         loss += sum(mlp.losses)
    #
    #         # Get gradients of weights wrt the loss.
    #         gradients = tape.gradient(loss, mlp.trainable_weights)
    #
    #     # Update the weights of our linear layer.
    #     optimizer.apply_gradients(zip(gradients, mlp.trainable_weights))
    #
    #     # Logging.
    #     if step % 100 == 0:
    #         print("Step:", step, "Loss:", float(loss))
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    # ]
    # VGG16.compile(
    #     optimizer=keras.optimizers.Adam(1e-3),
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"],
    # )
    # history=VGG16.fit(
    #     x_train4D_normalize, y_trainOnehot,  batch_size=32, epochs=25, validation_data=(x_test4D_normalize,y_testOnehot)
    # )
    # training_loss=history.history["loss"]
    # train_acc=history.history["accuracy"]
    # test_loss=history.history["val_loss"]
    # test_acc=history.history["val_accuracy"]
    # epoch_count=range(1,len(training_loss)+1)
    # plt.plot(epoch_count,training_loss,'r--')
    # plt.plot(epoch_count,test_loss,'b--')
    # plt.plot(epoch_count,train_acc,'r--')
    # plt.plot(epoch_count,test_acc,'b--')
    # plt.legend(["Training_loss","Test_loss","train_acc","test_acc"])
    # plt.xlabel("Epoch")
    # plt.ylabel("loss")
    # plt.show()
def trainmodel(model,savepath,mode):
    if not os.path.exists('./savedmodel/'+mode+'/'+savepath):
        os.makedirs('./savedmodel/'+mode+'/'+savepath)
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
        'D:/dataset/'+mode+'/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        'D:/dataset/'+mode+'/val',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')
    earlyStopping = EarlyStopping(monitor='val_loss',min_delta=0.01, patience=6, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('./savedmodel/'+mode+'/'+savepath+'/save_at_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, epsilon=1e-4, mode='min')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    # Train model on dataset
    history=model.fit_generator(
                                generator=train_generator,
                                validation_data=validation_generator,
                                steps_per_epoch=400,
                        epochs=80,
                        workers=8,
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
    plt.savefig('./savedmodel/'+mode+'/' + savepath + '/loss.jpg')
    plt.show()
    # X,y=validation_generator.__getitem__(1)
    plt.plot(epoch_count,train_acc,'r--')
    plt.plot(epoch_count,test_acc,'b--')
    plt.legend(["train_acc","test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("acc")
    plt.savefig('./savedmodel/'+mode+'/'+savepath+'/acc.jpg')
    plt.show()
# testmodel, need a parameter model, and weights
def testmodel(model=make_VGG16_model(input_shape=(224, 224, 3,), num_classes=4),Weights='./savedmodel/full/VGG16/save_at_21.h5',mode="JPG",which=""):
    model.load_weights(Weights)
    if mode=="Segmentation"or mode=="JPG":
        root_for_covid = 'D:/dataset/Covid/'
        rootpath = 'D:/dataset/Other/'
        generate_dataset = (
            lambda split: [root_for_covid + split, rootpath + split + 'NORMAL/',rootpath + split + 'PNEUMONIA/', rootpath + split + 'VIRUS/'])
        # testset path
        testset_filelist = generate_dataset('test/')
        X = np.empty((2110, 224,224, 3))
        y = np.empty((2110), dtype=int)
        partition_test,labels_test=generate_partitionandlabel(testset_filelist,3000)
        for ID,i in zip(partition_test,range(len(partition_test))):
            y[i]=labels_test[ID]
            if mode=="Segmentation":
                if labels_test[ID] == 0:
                    X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/COVID/' + ID + '.png'), (224, 224))/255
                else:
                    if labels_test[ID] == 1:
                        X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/NORMAL/' + ID + '.png'), (224, 224))/255
                    elif labels_test[ID] == 2:
                        X[i,]=cv2.resize(cv2.imread('D:/dataset/Segmentation/test/PNEUMONIA/' + ID + '.png'), (224, 224))/255
                    else:
                        X[i,] = cv2.resize(cv2.imread('D:/dataset/Segmentation/test/VIRUS/' + ID + '.png'), (224, 224))/255
            else:
                if labels_test[ID] == 0:
                    X[i,] = cv2.resize(cv2.imread('D:/dataset/JPG/test/COVID/' + ID + '.png'), (224, 224))/255
                else:
                    if labels_test[ID] == 1:
                        X[i,] = cv2.resize(cv2.imread('D:/dataset/JPG/test/NORMAL/' + ID + ''), (224, 224))/255
                    elif labels_test[ID] == 2:
                        X[i,] = cv2.resize(cv2.imread('D:/dataset/JPG/test/PNEUMONIA/' + ID + ''), (224, 224))/255
                    else:
                        X[i,] = cv2.resize(cv2.imread('D:/dataset/JPG/test/VIRUS/' + ID + ''), (224, 224))/255
    elif mode=="Segmented":
        X = np.empty((2110, 224,224, 3))
        y = np.empty((2110), dtype=int)
    elif mode=="Cleaned":
        X = np.empty((96, 224, 224, 3))
        y = np.empty((96), dtype=int)
        for root, dirs, files in os.walk('D:/dataset/external/'+mode+'/'):
            for i, f in zip(tqdm(range(len(files))), files):

                y[i]=0
                print(root+f)
                X[i,] = cv2.resize(cv2.imread(root + f), (224, 224)) / 255


    Y=keras.utils.to_categorical(y, num_classes=4)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    Score=model.evaluate(X,Y,verbose=0)
    print("Test loss",Score[0])
    print("Test accuracy",Score[1])
    predictions = model.predict(X)
    print(predictions)
    if mode=="Segmentation"or mode=="JPG":
        predictions = model.predict(X)

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
                 label='Micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        # plt.plot(fpr["macro"], tpr["macro"],
        #          label='macro-average ROC curve (area = {0:0.2f})'
        #                ''.format(roc_auc["macro"]),
        #          color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue',"black"])
        Labels=['COVID-19', 'Bacteria', 'Virus', 'Normal']
        print(roc_auc)
        for i, color in zip(range(4), colors):
            print(i)
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of '+Labels[i]+''+' (area = {1:0.2f})'.format(i,roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(''+which+'')
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
if __name__ == '__main__':
    # testmodel()
    # testmodel(model=make_VGG16_model(input_shape=(224, 224, 3,), num_classes=4),Weights='./savedmodel/VGG16/save_at_21.h5')
    # make_VGG19_model(input_shape=(224, 224,3,),num_classes=4)
    # trainmodelwithnpy(model=make_model(name="VGG19", input_shape=(224, 224, 3,), num_classes=4), savepath="VGG19")
    # for mode in ["Segmentation","JPG"]:
    #     for name in ["ResNet50", "MobileNet","VGG19", "InceptionV3", "VGG16",]:
    # #         # , "MobileNet""VGG19", "InceptionV3", "ResNet50"]
    #         trainmodel(model=make_model(name=name,input_shape=(224, 224,3,),num_classes=4), savepath=name,mode=mode)
    # for mode in ["Segmentation"]:
    #     for name in ["ResNet50"]:
    # #         # , "MobileNet""VGG19", "InceptionV3", "ResNet50"]
    #         trainmodel(model=make_model(name=name,input_shape=(224, 224,3,),num_classes=4), savepath=name,mode=mode)
    # for mode in ["Segmentation"]:
    #     for name in ["VGG161","InceptionResNetV2","DenseNet121"]:
    #         trainmodel(model=make_model(name=name,input_shape=(224, 224,3,),num_classes=4), savepath=name,mode=mode)

    # modeldict={"VGG161":13,"InceptionResNetV2":5,"DenseNet121":9,"ResNet50":11,"VGG16":19,"MobileNet":13,"InceptionV3":13,"VGG19":8}
    # for name in ["VGG161","InceptionResNetV2","DenseNet121","ResNet50", "MobileNet", "VGG19", "InceptionV3", "VGG16"]:
    #     testmodel(model = make_model(name=name,input_shape=(224, 224, 3,), num_classes=4),Weights='./savedmodel/Segmentation/'+name+'/save_at_'+str(modeldict[name])+'.h5',mode="Segmentation",which=name)

    testmodel(model=make_model(name="VGG19", input_shape=(224, 224, 3,), num_classes=4),
          Weights='./savedmodel/Segmentation/VGG19/save_at_8.h5', mode="Segmentation",
          which="VGG19")
