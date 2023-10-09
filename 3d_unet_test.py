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
from keras.utils.np_utils import normalize
from keras.models import load_model
from tifffile import imsave


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


dependencies = {
    'dice_coefficient': dice_coefficient,
    'dice_coefficient_loss': dice_coefficient_loss
}


my_model = load_model(r'D:\Vinay\3D U-Net\model-3dunet-mouse_heart.h5', custom_objects = dependencies)


image = io.imread(r'D:\Vinay\sandstone_data_for_ML\sandstone_data_for_ML\data_for_3D_Unet\448_images_512x512.tif') #2048x512x512
print(image.shape)

# # for x in range(64,2048,64):
img_patches = patchify(image, (32, 32, 32), step=32)  #Step=64 for 64 patches means no overlap
print(img_patches.shape)
# assert img_patches.shape == (4, 4, 4, 64, 64, 64)
# reconstructed_image = unpatchify(img_patches, image.shape)

# assert (reconstructed_image == image).all()
# print(reconstructed_image.shape)
# reconstructed_image=reconstructed_image.astype(np.uint8)
# print(reconstructed_image.dtype)

# # #Now save it as segmented volume.
# from tifffile import imsave
# imsave('D:/Vinay/3D U-Net/segmented.tif', reconstructed_image)


predicted_patches = []
# cnt = 0
for i in range(img_patches.shape[0]):
    for j in range(img_patches.shape[1]):
        for k in range(img_patches.shape[2]):
            single_patch = img_patches[i,j,k, :,:,:]
            single_patch_3ch = np.stack((single_patch,)*3, axis=-1) #make rgb
            single_patch_3ch = normalize(single_patch_3ch, axis=1)
            single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0) #expand dimensions
            single_patch_prediction = my_model.predict(single_patch_3ch_input)
            single_patch_prediction_argmax = np.argmax(single_patch_prediction, axis=4)[0,:,:,:]
            predicted_patches.append(single_patch_prediction_argmax)





#Convert list to numpy array
predicted_patches = np.array(predicted_patches)
print(predicted_patches.shape)



#Reshape to the shape we had after patchifying
predicted_patches_reshaped = np.reshape(predicted_patches, 
                                        (img_patches.shape[0], img_patches.shape[1], img_patches.shape[2],
                                         img_patches.shape[3], img_patches.shape[4], img_patches.shape[5]) )
print(predicted_patches_reshaped.shape)



plt.figure()
plt.subplot(121)
plt.imshow(img_patches[1,2,1,:,:,12])
plt.subplot(122)
plt.imshow(predicted_patches_reshaped[1,2,1,:,:,12])

plt.show()

# Repach individual patches into the orginal volume shape
reconstructed_image = unpatchify(predicted_patches_reshaped, image.shape)
print(reconstructed_image.shape)

print(reconstructed_image.dtype)

#Convert to uint8 so we can open image in most image viewing software packages
reconstructed_image=reconstructed_image.astype(np.uint8)
print(reconstructed_image.dtype)

#Now save it as segmented volume.

imsave('D:/Vinay/3D U-Net/example_segmented.tif', reconstructed_image)


