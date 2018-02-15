
from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np

import six
import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout, Flatten

from keras.layers import add
from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from keras.regularizers import l2
from keras import backend as K

from keras import optimizers

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler

BN_AXIS = 3

def residual_block(input_tensor, conv_kernel_size, filters, stage, block, downsample=False):

    '''
    enforce len(filters) == 3 when downsample is True.
    We need to know the number of filters to use in Conv layers for downsampled input
    '''
    if downsample:
        assert len(filters) == 3, "Expecting three filter sizes"
        nb_filter1, nb_filter2, nb_filter3 = filters
    else:
        nb_filter1, nb_filter2 = filters

    conv_base_name = 'conv_stage-' + str(stage) + '-block-' + block
    bn_base_name = 'bnorm_stage-' + str(stage) + '-block-' + block
    activation_base_name = 'actv_stage-' + str(stage) + '-block-' + block

    '''
    if downsample is True, then the first Conv2D layer in this residual 
    block must perform downsampling
    '''
    default_stride = (1, 1)
    first_layer_stride = default_stride
    if downsample:
        first_layer_stride = (2, 2)

    x = Conv2D(filters=nb_filter1, 
            kernel_size=(conv_kernel_size, conv_kernel_size), 
            padding='same', 
            strides=first_layer_stride, 
            name=conv_base_name + '-path-2a',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4))(input_tensor)

    x = BatchNormalization(axis=BN_AXIS, name=bn_base_name + '-path-2a')(x)
    x = Activation('relu', name=activation_base_name + '-path-2a')(x)

    x = Conv2D(filters=nb_filter2, 
            kernel_size=(conv_kernel_size, conv_kernel_size), 
            padding='same', 
            strides=default_stride, 
            name=conv_base_name + '-path-2b',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(axis=BN_AXIS, name=bn_base_name + '-path-2b')(x)

    '''
    if this is the case of a downsampling residual block, do a merge with a 1x1 conv filter
    with (2, 2)-strided/downsampled input_tensor
    '''
    shortcut = input_tensor
    if downsample:
        shortcut = Conv2D(filters=nb_filter3, 
                kernel_size=(1,1), 
                padding='same', 
                strides=first_layer_stride, 
                name=conv_base_name + '_mainline-1',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))(input_tensor)
        shortcut = BatchNormalization(axis=BN_AXIS, name=bn_base_name + '_mainline-1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu', name=activation_base_name + '_mainline-1')(x)

    return x


if __name__=='__main__':
    batch_size = 64
    num_classes = 10
    num_epochs = 200
    img_rows = img_cols = 32
    img_channels = 3

    ''' The data, shuffled and split between train and test sets: '''
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    ''' Convert class vectors to binary class matrices. '''
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)


    ''' 
    normalize inputs: subtract mean and divide by standard deviation 
    -- obtain the mean of each channel, and subtract from all images in train and test
    -- divide each channel by the standard deviation

    Resnet authors He et al. only performed per-pixel mean subtraction from cifar10 images
    No other normalization was performed.
    Perhaps the reason for this is the BatchNormalization layers in the network help 
    regularization and training.
    '''
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image

    ''' 
    reason for using the 'training' mean is to enable uniform transformation 
    of the data an to account for cases where the entire testset is not available 
    at once for calculating mean test image 
    '''
    X_test -= mean_image 


    '''
    We will be using Tensorflow backend for Keras
    This means dim_ordering needs to be provided as (img_rows, img_cols, img_channels)
    Just cross-check as to what cifar10.load_data() loads it as
    Here is a clear explanation: http://www.codesofinterest.com/2017/05/image-data-format-vs-image-dim-ordering-keras-v2.html

    In ~/keras/keras.json, image_dim_ordering is set to 'tf' or 'th'
    '''


    '''
    Residual Block Architecture
    (based on Deep Residual Learning for Image Recognition: http://arxiv.org/abs/1512.03385)

    RESNET18

    INPUT ---> CONV --> BATCHNORM --> RELU --> CONV --> BATCHNORM -- ADD -->  RELU --> OUTPUT
        |
        |                                                             ^
        |                                                             |
        +-------------------------------------------------------------+

    In case there is a problem, input_shape can be manipulated using numpy slicing below 
    Shape of X_train is (num_images, img_rows, img_cols, img_channels)
    '''
    #input_shape = X_train.shape[1:][-1:] + X_train.shape[1:][:-1]

    '''
    He et al's paper: Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
    uses slightly different setup for CIFAR10 dataset.
    '''

    img_input = Input(shape=X_train.shape[1:], name='cifar10')

    conv0 = Conv2D(filters=16, 
            kernel_size=(3, 3), 
            padding='same', 
            strides=(1, 1), 
            name='input_conv_1',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(1e-4))(img_input)

    bn0 = BatchNormalization(axis=BN_AXIS, name='input_bnorm1')(conv0)
    x = Activation('relu', name='input_actv_1')(bn0)

    x = residual_block(x, 3, [16, 16], 2, block='a') ### 2 layers
    x = residual_block(x, 3, [16, 16], 2, block='b') ### 2 layers
    x = residual_block(x, 3, [16, 16], 2, block='c') ### 2 layers

    x = residual_block(x, 3, [32, 32, 32], 3, block='a', downsample=True) ### 2 layers
    x = residual_block(x, 3, [32, 32], 3, block='b') ### 2 layers
    x = residual_block(x, 3, [32, 32], 3, block='c') ### 2 layers

    x = residual_block(x, 3, [64, 64, 64], 4, block='a', downsample=True) #### 2 layers
    x = residual_block(x, 3, [64, 64], 4, block='b') ### 2 layers
    x = residual_block(x, 3, [64, 64], 4, block='c') ### 2 layers

    x = AveragePooling2D((8, 8), name='avg_pooling')(x)

    x = Flatten(name='flatten_for_fc')(x)
    ''' x = Dense(1000, name='fc1000')(x) '''
    ''' x = Activation('relu', name='activation_fc1000')(x) '''
    ''' x = Dropout(0.5)(x) '''
    x = Dense(num_classes, name='fc10')(x)
    x = Activation('softmax', name='softmax_out')(x)

    model = Model(img_input, x)

    ''' using two optimizers:
        (1) adam
        (2) SGD (nesterov)
        (3) nadam (Nesterov + ADAM optimizer: http://cs229.stanford.edu/proj2015/054_report.pdf)
    '''
    def cifar10_lr_schedule(epoch):
        if epoch <= 10:
            lr = 0.01
        elif 10 < epoch <= 82:
            lr = 0.1
        elif 82 < epoch <= 123:
            lr = 0.01
        else:
            lr = 0.001
        return lr

    scheduled_lr = LearningRateScheduler(cifar10_lr_schedule)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
    sgd = optimizers.Adam(lr=3e-4)
    # sgd = optimizers.Adam()

    ### callbacks
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    # csv_logger = CSVLogger('train_SGD_lr=0.01_momentum=0.9_nesterov.csv')
    csv_logger = CSVLogger('train_adam_lr=3e-4.csv')
    # csv_logger = CSVLogger('train_adam_lr=default.csv')


    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(X_test, y_test),
              shuffle=True,
              callbacks=[lr_reducer, csv_logger, early_stopper])

