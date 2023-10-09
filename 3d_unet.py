
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate
from unittest.mock import patch
from Multiclass_3d_unet_model import Unet 
import tensorflow as tf
import keras
from skimage import io
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.metrics import MeanIoU
from PIL import Image
import cv2
import random
import tifffile
from keras.preprocessing.image import ImageDataGenerator
# from segmentation_models.metrics import iou_score
# import segmentation_models as sm
from sklearn.utils import class_weight
# from focal_loss import SparseCategoricalFocalLoss
# from augmented import generator
# from loss_functions import focal_loss, sym_unified_focal_loss
# import elasticdeform
# import tensorflow as tf

# with tf.device("/gpu:0"):

def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    # x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    # x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes, n_filters=64):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, n_filters)
    s2, p2 = encoder_block(p1, n_filters*2)
    s3, p3 = encoder_block(p2, n_filters*4)
    s4, p4 = encoder_block(p3, n_filters*8)

    b1 = conv_block(p4, n_filters*16) #Bridge

    d1 = decoder_block(b1, s4, n_filters*8)
    d2 = decoder_block(d1, s3, n_filters*4)
    d3 = decoder_block(d2, s2, n_filters*2)
    d4 = decoder_block(d3, s1, n_filters)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model


# my_model = build_unet((64,64,64,3), n_classes=4)
IMG_HEIGHT = 16
IMG_WIDTH = 16
IMG_DEPTH = 16
IMG_CHANNELS = 3
# n_filters = 16


n_classes=2
step_size = 16

#Load input images and masks. 
#Here we load 512x512x512 pixel volume. We will break it into patches of 64x64x64 for training. 
image = io.imread(r'./raw 512x512.tif') #2048x512x512
print(image.shape)

# # for x in range(64,2048,64):
img_patches = patchify(image, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)  #Step=64 for 64 patches means no overlap
print(img_patches.shape)

mask = io.imread(r'./predicted binary 512x512.tif')
print(mask.shape)

mask_patches = patchify(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), step=step_size)
print(mask_patches.shape)
print(np.unique(mask_patches))




# random_num = random.randint(0,15)
# print(random_num)
# plt.figure()
# plt.subplot(121)
# plt.imshow(img_patches[2,2,1,:,:,random_num], cmap='gray')     
# plt.subplot(122)
# plt.imshow(mask_patches[2,2,1,:,:,random_num])

# plt.show()

input_img = np.reshape(img_patches, (-1, img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]))
input_mask = np.reshape(mask_patches, (-1, mask_patches.shape[3], mask_patches.shape[4], mask_patches.shape[5]))

print(input_img.shape) 
print(input_mask.shape)# n_patches, x, y, z

# Convert grey image to 3 channels by copying channel 3 times.
# We do this as our unet model expects 3 channel input. 
from keras.utils.np_utils import normalize

train_img = np.stack((input_img,)*3, axis=-1)
# train_img = train_img / 255.
train_img = normalize(train_img, axis=1)
print(train_img.shape)
train_mask = np.expand_dims(input_mask, axis=4)
# train_mask = train_mask / 255.
print(train_mask.shape)
print("Class values in the dataset are ... ", np.unique(train_mask))
train_mask_cat = to_categorical(train_mask, num_classes=n_classes)
print(train_mask_cat.shape)
# print(np.unique(train_mask_cat))







X_train, X_test, y_train, y_test = train_test_split(train_img, train_mask_cat, test_size = 0.2, random_state=42)

print(f'X_train:{X_train.shape}')
print(f'y_train:{y_train.shape}')



def jacard_coef(y_true, y_pred):
    y_true_f = np.asarray(y_true).astype(bool)
    y_pred_f = np.asarray(y_pred).astype(bool)
    intersection = np.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + 1.0)
  
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 0.0001
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# def focal_loss(alpha=0.25, gamma=2):
#     def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
#         targets = tf.cast(targets, tf.float32)
#         weight_a = alpha * (1 - y_pred) ** gamma * targets
#         weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    
#         return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 

#     def loss(y_true, logits):
#         y_pred = tf.math.sigmoid(logits)
#         loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

#         return tf.reduce_mean(loss)

#     return loss
  
  
  

model = build_unet((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), n_classes=n_classes, n_filters=16)

#Unet(n_filters, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, n_classes)
print(model.input_shape)
print(model.output_shape)

optimizer = keras.optimizers.Adam(lr=0.0001)
# decay_steps = 1000
# lr_decayed_fn = keras.optimizers.schedules.CosineDecay(0.0005, decay_steps)



# model = get_model()
model.compile(optimizer='adam', loss=dice_coefficient_loss, metrics=['accuracy',dice_coefficient])
model.summary()


# X_deformed = elasticdeform.deform_random_grid(X_train, axis=(2,3))
# Y_deformed = elasticdeform.deform_random_grid(y_train, axis=(2,3))


# seed = 42
# image_aug = generator.customImageDataGenerator(
# 			rotation_range = 45,
#             shear_range=20,
#             horizontal_flip=True,
#             vertical_flip=True,
         
           
# )

# mask_aug = generator.customImageDataGenerator(
# 			rotation_range = 45,
#             shear_range=20,
#             horizontal_flip=True,
#             vertical_flip=True,
#             #elastic_deform=True
            
# )

# X_train_datagen = image_aug.flow(X_deformed, batch_size= 2, seed=seed) # set equal seed number
# Y_train_datagen = mask_aug.flow(Y_deformed, batch_size = 2, seed=seed) # set equal seed number
# train_generator = zip(X_train_datagen, Y_train_datagen)

# X_validation_datagen = image_aug.flow(X_test, batch_size= 2, seed=seed) # set equal seed number
# Y_validation_datagen= mask_aug.flow(y_test, batch_size= 2, seed=seed) # set equal seed number
# validation_generator = zip(X_validation_datagen, Y_validation_datagen)








callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=r"D:/Vinay/3D U-Net/logs/{}".format('model-3dunet-mouseheart')),
    ModelCheckpoint('D:/Vinay/3D U-Net/model-3dunet-mouse_heart.h5', verbose=1, save_best_only=True)
]


# print(X_train_datagen.shape)
# print(train_generator.shape)


# history = model.fit(
#   train_generator,
#   steps_per_epoch = len(y_train) // 2,
#   epochs = 100,
#   validation_data = validation_generator,
#   validation_steps = len(y_test) // 2,
#   callbacks = callbacks
# )

history = model.fit(X_train, y_train, 
                    batch_size = 1, 
                    verbose=1, 
                    epochs=10, 
                    validation_data=(X_test, y_test), 
                    callbacks = callbacks)



#Evaluate the model
_, acc, *accc = model.evaluate(X_test, y_test)
print("Accuracy is = ", (acc * 100.0), "%")



loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_coefficient']
val_acc = history.history['val_dice_coefficient']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=4)
y_test_argmax=np.argmax(y_test, axis=4)

print(y_pred_argmax.shape)
print(y_test_argmax.shape)
print(np.unique(y_pred_argmax))

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


import random
test_img_number = random.randint(0, len(X_test))
print(test_img_number)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
# print(test_img.shape)
# print(ground_truth.shape)

test_img_input=np.expand_dims(test_img, 0)
# test_img_input1 = preprocess_input(test_img_input)

test_pred = model.predict(test_img_input)
test_prediction = np.argmax(test_pred, axis=4)[0,:,:,:]
# print(test_prediction)
print(test_prediction.shape)

ground_truth_argmax = np.argmax(ground_truth, axis=3)
print(ground_truth_argmax.shape)


slice_frame = random.randint(0, ground_truth_argmax.shape[2]-1)
print(slice_frame)
plt.figure()
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[slice_frame,:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth_argmax[slice_frame,:,:])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction[slice_frame,:,:])
plt.show()




#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
print(values.shape)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
# class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
# print("IoU for class3 is: ", class3_IoU)
# print("IoU for class4 is: ", class4_IoU)


# # # #Seperate each channel/segment to be combined as multiple channels.
# # num_segments=4
# # segm0 = (reconstructed_image == 0)
# # segm1 = (reconstructed_image == 1)
# # segm2 = (reconstructed_image == 2)
# # segm3 = (reconstructed_image == 3)

# # final = np.empty((reconstructed_image.shape[0], reconstructed_image.shape[1], reconstructed_image.shape[2], num_segments))
# # final[:,:,:,0] = segm0
# # final[:,:,:,1] = segm1
# # final[:,:,:,2] = segm2
# # final[:,:,:,3] = segm3

# # #Use APEER OMETIFF library to read and write multidimensional images
# # # !pip install apeer-ometiff-library

# # from apeer_ometiff_library import io

# # # Expand image array to 5D of order (T, Z, C, X, Y)
# # # This is the convention for OMETIFF format as written by APEER library
# # final = np.expand_dims(final, axis=0)
# # final=np.swapaxes(final, 2, 4)

# # final = final.astype(np.int8)

# # print("Shape of the segmented volume is: T, Z, C, X, Y ", final.shape)
# # print(final.dtype)

# # # Write dataset as multi-dimensional OMETIFF *image*
# # io.write_ometiff("D:/Vinay/3D U-Net/segmented_multi_channel.ome.tiff", final)