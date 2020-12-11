import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import pydot
import graphviz
from tensorflow_core.python.keras.utils import to_categorical
from matplotlib import pyplot as plt
def make_model(input_shape,num_classes):
    inputs=keras.Input(shape=input_shape)
    x = layers.Conv2D(input_shape=input_shape,kernel_size=(3,3),filters=64,strides=1,padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=128, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=256, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x )
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Conv2D(input_shape=input_shape, kernel_size=(3, 3), filters=512, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(4096)(x)
    x = layers.Activation('relu')(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    return  keras.Model(inputs, outputs)
(x_train, y_train),(x_test, y_test)= tf.keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# for each in range(x_train.shape[0]):
#     a.append(cv2.resize(x_train[each], (224, 224)))
# for each in range(x_test.shape[0]):
#     b.append(cv2.resize(x_test[each], (224, 224)))
# x_train = np.expand_dims(np.array(a), -1)
# x_test = np.expand_dims(np.array(b), -1)
x_train4D = [cv2.cvtColor(cv2.resize(i,(32,32)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train4D = np.concatenate([arr[np.newaxis] for arr in x_train4D]).astype('float32')
x_test4D = [cv2.cvtColor(cv2.resize(i,(32,32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test4D = np.concatenate([arr[np.newaxis] for arr in x_test4D]).astype('float32')
# dataset = tf.data.Dataset.from_tensor_slices(
#     (x_train.reshape(60000, 28,28,1).astype("float32") / 255, y_train)
# )
# dataset = dataset.shuffle(buffer_size=1024).batch(64)
x_test4D_normalize=x_test4D/255
x_train4D_normalize=x_train4D/255

"""one_hot encoding"""
y_trainOnehot=to_categorical(y_train)
y_testOnehot=to_categorical(y_test)
VGG16 = make_model(input_shape=(32,32,3,), num_classes=10)
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
epochs = 2

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
VGG16.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
history=VGG16.fit(
    x_train4D_normalize, y_trainOnehot,  batch_size=64, epochs=2, validation_data=(x_test4D_normalize,y_testOnehot)
)
training_loss=history.history["loss"]
train_acc=history.history["accuracy"]
test_loss=history.history["val_loss"]
test_acc=history.history["val_accuracy"]
epoch_count=range(1,len(training_loss)+1)
plt.plot(epoch_count,training_loss,'r--')
plt.plot(epoch_count,test_loss,'b--')
plt.plot(epoch_count,train_acc,'r--')
plt.plot(epoch_count,test_acc,'b--')
plt.legend(["Training_loss","Test_loss","train_acc","test_acc"])
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.show()