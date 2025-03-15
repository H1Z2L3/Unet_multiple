# We significanlty reduce the number of layers and features at each
# layer to save GPU memory and computation but still preserve high
# performace in fault segmentation.

import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

#######################################################################################################################
def unet(pretrained_weights = None,input_size = (None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv4], axis=-1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (2,2), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (2,2), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2,2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (2,2), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (2,2), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(32, (3, 3), activation='tanh', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1))(conv9)#
    # conv10 = Conv2D(1, (1, 1),activation='tanh')(conv9)



    # conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)# 
    # conv8 = Conv2D(1, (1, 1), activation='softmax')(conv7) #
    # conv8 = Conv2D(1, (1, 1), activation='relu')(conv7)#
    # conv8 = Conv2D(1, (1, 1), activation='tanh')(conv7)
    # conv8 = Conv2D(1, (1, 1))(conv7)#

    model = Model(inputs=[inputs], outputs=[conv10])
    model.summary()
    #model.compile(optimizer = Adam(lr = 1e-4),
    #    loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model
#######################################################################################################################
def unet1(pretrained_weights = None,input_size = (None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3,3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3,3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv2D(128, (3,3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3,3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv2D(256, (3,3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv2D(256, (3,3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256, (3,3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv2D(512, (3,3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4= LeakyReLU(alpha=0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv4], axis=-1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (2,2), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(512, (3,3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (2,2), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(256, (3,3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(128, (3,3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)

    up8 = concatenate([UpSampling2D(size=(2,2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (2,2), padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Conv2D(128, (3,3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Conv2D(64, (3,3), padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.2)(conv8)

    up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (2,2), padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.2)(conv9)
    conv9 = Conv2D(64, (3,3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.2)(conv9)
    # conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.2)(conv9)

    # conv10 = Conv2D(1, (1, 1))(conv9)
    conv10 = Conv2D(1, (1, 1),activation='tanh')(conv9)



    # conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='softmax')(conv7) 
    # conv8 = Conv2D(1, (1, 1), activation='relu')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='tanh')(conv7)
    # conv8 = Conv2D(1, (1, 1))(conv7)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.summary()
    #model.compile(optimizer = Adam(lr = 1e-4),
    #    loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model

#######################################################################################################################
def unet2(pretrained_weights = None,input_size = (None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv4], axis=-1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (2,2), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (2,2), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2,2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (2,2), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (2,2), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(32, (3, 3), activation='tanh', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1))(conv9)
    # conv10 = Conv2D(1, (1, 1),activation='tanh')(conv9)



    # conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='softmax')(conv7) 
    # conv8 = Conv2D(1, (1, 1), activation='relu')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='tanh')(conv7)
    # conv8 = Conv2D(1, (1, 1))(conv7)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.summary()
    #model.compile(optimizer = Adam(lr = 1e-4),
    #    loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model

def unet3(pretrained_weights = None,input_size = (None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3,3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv2D(128, (3,3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, (3,3), padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv2D(256, (3,3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Conv2D(256, (3,3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512, (3,3), padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    conv4 = Conv2D(512, (3,3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.2)(conv4)
    # conv4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)

    # conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = LeakyReLU(alpha=0.2)(conv5)
    # conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = LeakyReLU(alpha=0.2)(conv5)
    # conv5 = Conv2D(1024, (3, 3), padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)
    # conv5 = LeakyReLU(alpha=0.2)(conv5)

    up5 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
    # up5 = Dropout(0.5)(up5)
    conv5 = Conv2D(256, (2,2), padding='same')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv2D(256, (3,3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)
    conv5 = Conv2D(256, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(128, (2,2), padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(128, (3,3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(128, (3,3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(64, (2,2), padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(64, (3,3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(64, (3,3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(8, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Conv2D(2, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.2)(conv7)

    # up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
    # conv9 = Conv2D(64, (2,2), padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = LeakyReLU(alpha=0.2)(conv9)
    # conv9 = Conv2D(64, (3,3), padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = LeakyReLU(alpha=0.2)(conv9)
    # # conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)
    # conv9 = LeakyReLU(alpha=0.2)(conv9)

    conv8 = Conv2D(1, (1, 1))(conv7)
    # conv8 = Conv2D(1, (1, 1),activation='tanh')(conv7)



    # conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='softmax')(conv7) 
    # conv8 = Conv2D(1, (1, 1), activation='relu')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='tanh')(conv7)
    # conv8 = Conv2D(1, (1, 1))(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    model.summary()
    #model.compile(optimizer = Adam(lr = 1e-4),
    #    loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model
########################################################################################################################
def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits,
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
########################################################################################################################
def unet4(pretrained_weights = None,input_size = (None,None,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    # conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3,3), activation='relu', padding='same')(conv2)
    # conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    # conv3 = Conv2D(256, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3,3), activation='relu', padding='same')(conv4)
    # conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    #
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    # conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2,2))(conv5), conv4], axis=-1)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(512, (2,2), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3,3), activation='relu', padding='same')(conv6)
    # conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2,2))(conv4), conv3], axis=-1)
    conv7 = Conv2D(256, (2,2), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3,3), activation='relu', padding='same')(conv7)
    # conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2,2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (2,2), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3,3), activation='relu', padding='same')(conv8)
    # conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2,2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (2,2), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(64, (3,3), activation='relu', padding='same')(conv9)
    # conv9 = Conv2D(32, (3, 3), activation='tanh', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1))(conv9)
    # conv10 = Conv2D(1, (1, 1),activation='tanh')(conv9)



    # conv8 = Conv2D(1, (1,1), activation='sigmoid')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='softmax')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='relu')(conv7)
    # conv8 = Conv2D(1, (1, 1), activation='tanh')(conv7)
    # conv8 = Conv2D(1, (1, 1))(conv7)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.summary()
    #model.compile(optimizer = Adam(lr = 1e-4),
    #    loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model
#######################################################################################################################
