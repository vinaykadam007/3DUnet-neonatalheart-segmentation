from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda, Activation
from keras import backend as K


# def unet(0., initial_features=128, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=4, patch_size = 512, IMAGE_HEIGHT=512, IMAGE_WIDTH=512):
#     inputs = Input(shape=(patch_size, IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
#     x = inputs
    
#     convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
#     #downstream
#     skips = {}
#     for level in range(n_levels):
#         for _ in range(n_blocks):
#             x = Conv3D(initial_features * 2 ** level, **convpars)(x)
#         if level < n_levels - 1:
#             skips[level] = x
#             x = MaxPooling3D(pooling_size)(x)
            
#     # upstream
#     for level in reversed(range(n_levels-1)):
#         x = Conv3DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
#         x = concatenate([x, skips[level]])
#         for _ in range(n_blocks):
#             x = Conv3D(initial_features * 2 ** level, **convpars)(x)
            
#     # output
#     activation = 'sigmoid' if out_channels == 1 else 'softmax'
#     x = Conv3D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
#     return Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

# def conv3d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
#     """Function to add 2 convolutional layers with the parameters passed to it"""
#     # first layer
#     x = Conv3D(filters = n_filters, kernel_size=kernel_size, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
#     # second layer
#     x = Conv3D(filters = n_filters, kernel_size=kernel_size, kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
#     if batchnorm:
#         x = BatchNormalization()(x)
#     x = Activation('relu')(x)
    
#     return x

# def Unet(n_filters = 16, dropout = 0.1, batchnorm = True, patch_size=64, IMG_HEIGHT=64, IMG_WIDTH=64, IMG_CHANNELS=1, n_classes = 4):
#     """Function to define the UNET Model"""
    
#     input_img = Input(shape=(patch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    
#     # Contracting Path
#     c1 = conv3d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
#     p1 = MaxPooling3D((2, 2, 2))(c1)
#     p1 = Dropout(dropout)(p1)
    
#     c2 = conv3d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
#     p2 = MaxPooling3D((2, 2, 2))(c2)
#     p2 = Dropout(dropout)(p2)
    
#     c3 = conv3d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
#     p3 = MaxPooling3D((2, 2, 2))(c3)
#     p3 = Dropout(dropout)(p3)
    
#     c4 = conv3d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#     p4 = MaxPooling3D((2, 2, 2))(c4)
#     p4 = Dropout(dropout)(p4)
    
#     c5 = conv3d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
#     # Expansive Path
#     u6 = Conv3DTranspose(n_filters * 8, (2, 2, 2), strides = 2, padding = 'same')(c5)
#     u6 = concatenate([u6, c4])
#     u6 = Dropout(dropout)(u6)
#     c6 = conv3d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
#     u7 = Conv3DTranspose(n_filters * 4, (2, 2, 2), strides = 2, padding = 'same')(c6)
#     u7 = concatenate([u7, c3])
#     u7 = Dropout(dropout)(u7)
#     c7 = conv3d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
#     u8 = Conv3DTranspose(n_filters * 2, (2, 2, 2), strides = 2, padding = 'same')(c7)
#     u8 = concatenate([u8, c2])
#     u8 = Dropout(dropout)(u8)
#     c8 = conv3d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
#     u9 = Conv3DTranspose(n_filters * 1, (2, 2, 2), strides = 2, padding = 'same')(c8)
#     u9 = concatenate([u9, c1])
#     u9 = Dropout(dropout)(u9)
#     c9 = conv3d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
#     outputs = Conv3D(n_classes, (1, 1, 1), activation='sigmoid')(c9)
#     model = Model(inputs=[input_img], outputs=[outputs])
#     return model



kernel_initializer =  'he_uniform' #Try others if you want


################################################################
def Unet(n_filters, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)
    
    c2 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
     
    c3 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)
     
    c4 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)
     
    c5 = Conv3D(n_filters*16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv3D(n_filters*16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    
    #Expansive path 
    u6 = Conv3DTranspose(n_filters*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv3D(n_filters*8, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)
    c6 = BatchNormalization()(c6)
     
    u7 = Conv3DTranspose(n_filters*4, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv3D(n_filters*4, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)
    c7 = BatchNormalization()(c7)
     
    u8 = Conv3DTranspose(n_filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(n_filters*2, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)
    c8 = BatchNormalization()(c8)
     
    u9 = Conv3DTranspose(n_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(n_filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)
    c9 = BatchNormalization()(c9)
    
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    # model.summary()
    
    return model



