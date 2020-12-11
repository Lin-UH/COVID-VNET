import heapq

import pydicom
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras
import sys
import cv2

import prepare_dataset
tf.compat.v1.disable_eager_execution()
def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
    for var, grad in zip(var_list, grads)]

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(filePath_covid):

    ds = pydicom.read_file(filePath_covid)
    pix = np.stack((cv2.resize(ds.pixel_array, (256, 256)),) * 3, axis=-1)
    # img = image.load_img(filePath_covid, target_size=(256, 256))
    # x = image.img_to_array(img)
    x = np.expand_dims(pix, axis=0)
    x=x/255
    # x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='block5_conv3'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == tensorflow.keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        model = prepare_dataset.make_model(input_shape=(256, 256, 3,), num_classes=4)

        model.load_weights('./newmodel.h5')

    return model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
    model = Sequential()
    model.add(input_model)

    nb_classes =4
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    # (anonymity function in python called by target_layer )
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    # 一般有两种方法，一种是直接定义类class然后继承Layer，一种是直接使用Lambda函数。
    loss = K.sum(model.layers[-1].output)
    # print(np.array([1]).shape)
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    grads = normalize(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (256, 256))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    # image -= np.min(image)
    # image = np.minimum(image, 255)
    image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    # cv2.imshow(winname="sd", mat=cam)
    # cv2.waitKey(0)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


# sys.argv[1]

# model = VGG16(weights='imagenet')




model = prepare_dataset.make_model(input_shape=(256,256,3,), num_classes=4)

model.load_weights('./newmodel.h5')





# print(predictions)
# predictions=np.random.normal(loc=0,scale=1,size=(1,1000))
# print(predictions)
# top_1 = decode_predictions(predictions)[0][0]
# # return (class_name, class_description, score)
# print('Predicted class:')
# print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))

# predicted_class = np.argmax(predictions)
# print(preprocessed_input.shape,predicted_class)
# cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, "block5_conv3")
# cv2.imwrite("gradcam.jpg", cam)
#
# register_gradient()
# guided_model = modify_backprop(model, 'GuidedBackProp')
# saliency_fn = compile_saliency_function(guided_model)
# saliency = saliency_fn([preprocessed_input, 0])
# gradcam = saliency[0] * heatmap[..., np.newaxis]
# cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
image_path = r'F:/Edge_download/data/dicom/new/dicom_archive_v2.tar/dicom_clean/9613200176504071786.dcm'
preprocessed_input = load_image(image_path)

register_gradient()
guided_model = modify_backprop(model, 'GuidedBackProp')
saliency_fn = compile_saliency_function(guided_model)
saliency = saliency_fn([preprocessed_input, 0])
gradcam = saliency[0].transpose(1, 2, 3, 0)
a = np.squeeze(gradcam)
cv2.imshow(r'Guided_BP', deprocess_image(a))
cv2.waitKey(0)
cv2.imwrite(r'.\imagenet_test\Guided_BP.jpg', deprocess_image(a))

pred = model.predict(preprocessed_input)
top1_idx, top2_idx, top3_idx= heapq.nlargest(3, range(len(pred[0])), pred[0].take)
# top_1 = decode_predictions(pred)[0][0]
# top_2 = decode_predictions(pred)[0][1]
# top_3 = decode_predictions(pred)[0][2]
# print('Predicted class:')
# print('%s (%s , %d) with probability %.2f' % (top_1[1], top_1[0], top1_idx, top_1[2]))
# print('%s (%s , %d) with probability %.2f' % (top_2[1], top_2[0], top2_idx, top_2[2]))
# print('%s (%s , %d) with probability %.2f' % (top_3[1], top_3[0], top3_idx, top_3[2]))
class_output = model.output[:, top1_idx]
# print(class_output)
last_conv_layer = model.get_layer("block5_pool")
grads = K.gradients(class_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([preprocessed_input])

for i in range(512):
    # print(conv_layer_output_value[:, :, i],pooled_grads_value[i])
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
print(heatmap)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

img = pydicom.read_file(image_path)
print(img.pixel_array.shape)
# img = np.stack((cv2.resize(img.pixel_array, (256, 256)),) * 3, axis=-1)
img = np.stack((cv2.resize(img.pixel_array, (256, 256)),) * 3, axis=-1)
cv2.imshow('heatmap', img)
cv2.waitKey(0)
# img = img_to_array(image)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
cv2.imwrite(r'.\imagenet_test\Heatmap.jpg', heatmap)
cv2.imshow('heatmap', heatmap)
cv2.waitKey(0)

heatmap2color = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
# heatmap2color = np.maximum(heatmap2color, 0)
# heatmap2color = heatmap2color / np.max(heatmap2color)


# heatmap = np.stack((cv2.resize(heatmap, (256, 256)),) * 3, axis=-1)
cv2.imshow('heatmap', heatmap)
cv2.waitKey(0)
print(img.shape)
print(heatmap2color.shape)
# grd_CAM = cv2.addWeighted(img, 0.6, heatmap2color, 0.4, 0,dtype=cv2.CV_64FC3)
# cv2.normalize(heatmap2color,heatmap2color,0,255,cv2.NORM_MINMAX)
# cv2.imwrite(r'.\imagenet_test\Grd-CAM.jpg', grd_CAM)
# cv2.imshow('Grd-CAM', heatmap2color)
# cv2.waitKey(0)
cam = np.float32(img) + np.float32(heatmap2color)
cam = 255 * cam / np.max(cam)
cv2.imshow('Grd-CAM', np.uint8(cam))
cv2.waitKey(0)

heatmap =cv2.imread(r'.\imagenet_test\Heatmap.jpg')
guided_CAM = saliency[0].transpose(1, 2, 3, 0) * heatmap[..., np.newaxis]
guided_CAM = deprocess_image(guided_CAM)
cv2.imwrite(r'.\imagenet_test\Guided-CAM.jpg', guided_CAM)
cv2.imshow('Guided-CAM', guided_CAM)
cv2.waitKey(0)


