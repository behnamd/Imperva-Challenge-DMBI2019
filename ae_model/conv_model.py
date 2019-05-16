import tensorflow as tf
import keras
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, Flatten, Reshape, UpSampling2D, \
    Conv2DTranspose, MaxPooling2D
from keras.initializers import he_normal
from keras import regularizers
from keras import objectives
from keras.models import Sequential, load_model
from keras.layers import *  # Dense, LSTM, Dropout, GRU, Conv2D, MaxPooling2D, Flatten, merge, Merge, UpSampling2D, \
from keras.models import Model
weight_decay = 0.0005


class HashModel:
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.stack_num = stack_num
        self.hash_bits = hash_bits

    # you have to overite this function for the inherrent class
    def net_loss(self, y_true, y_pred):
        return 0


def residual_block(x, shape, filter_type=''):
    output_filter_num = shape[1]
    if filter_type == 'increase':  # set the stride to (2, 2) is just like pooling
        first_stride = (2, 2)
    elif filter_type == 'decrease':
        x = UpSampling2D()(x)  # if filter is decrease, we Unsample the x first
        first_stride = (1, 1)
    else:
        first_stride = (1, 1)

    pre_bn = BatchNormalization()(x)
    pre_relu = Activation('relu')(pre_bn)

    conv_1 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=first_stride,  # if 'increase', change the feature map size here (pooling)
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(pre_relu)
    bn_1 = BatchNormalization()(conv_1)
    relu1 = Activation('relu')(bn_1)
    conv_2 = Conv2D(output_filter_num,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer=he_normal(),
                    kernel_regularizer=regularizers.l2(weight_decay)
                    )(relu1)
    if filter_type == 'increase':  # change the image size and channel from x to block
        projection = Conv2D(output_filter_num,
                            kernel_size=(1, 1),
                            strides=(2, 2),
                            padding='same',
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    elif filter_type == 'decrease':
        projection = Conv2D(output_filter_num,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay)
                            )(x)
        block = add([conv_2, projection])
    else:
        block = add([conv_2, x])
    return block


# abstract class for hash model
# you have to define the children class to inherate the __init__ and net_loss function,
# and name your hash_layer to 'hash_x'


class HashSupervisedAutoEncoderModel(HashModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma):
        HashModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # build the supervised autoencoder model
        self.img_input = Input(shape=(self.img_rows, self.img_cols, self.img_channels), name="img_input")

        x = Conv2D(filters=16,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   padding='same',
                   kernel_initializer=he_normal(),
                   kernel_regularizer=regularizers.l2(weight_decay),
                   )(self.img_input)

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = residual_block(x, [16, 32], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [16, 32])

        x = residual_block(x, [32, 64], filter_type='increase')
        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 64])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        shape_restore = x.get_shape().as_list()[1:4]
        units_restore = shape_restore[0] * shape_restore[1] * shape_restore[2]
        x = Flatten()(x)
        self.hash_x = Dense(hash_bits, activation='sigmoid', kernel_initializer=he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay), name="hash_x")(x)

        ## build the decoder model
        x = Dense(units_restore, activation='relu', kernel_initializer=he_normal(),
                  kernel_regularizer=regularizers.l2(weight_decay))(self.hash_x)

        x = Reshape((shape_restore[0], shape_restore[1], shape_restore[2]))(x)

        for _ in range(1, self.stack_num):
            x = residual_block(x, [64, 64])
        x = residual_block(x, [64, 32], filter_type='decrease')

        for _ in range(1, self.stack_num):
            x = residual_block(x, [32, 32])
        x = residual_block(x, [32, 16], filter_type='decrease')

        for _ in range(0, self.stack_num):
            x = residual_block(x, [16, 16])

        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        self.y_decoded = Conv2D(filters=self.img_channels,
                                activation='sigmoid',
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=he_normal(),
                                kernel_regularizer=regularizers.l2(weight_decay),
                                name='y_decoded'
                                )(x)

        self.y_predict = Dense(self.num_classes, activation='softmax', kernel_initializer=he_normal(),
                               kernel_regularizer=regularizers.l2(weight_decay), name='y_predict')(self.hash_x)




    def net_loss(self, y_true, y_pred):
        supervised_loss = 0  # objectives.categorical_crossentropy(y_true, y_pred)  # we calculate supervised loss outside
        binary_loss = - tf.reduce_mean(tf.square(self.hash_x - 0.5))
        balance_loss = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x, 0) - 0.5))
        decoded_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return supervised_loss + self.alpha * binary_loss + self.beta * balance_loss + self.gamma * decoded_loss





''''
class HashSupervisedPairedAutoEncoderModel(HashSupervisedAutoEncoderModel):
    def __init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma,delta):
        HashSupervisedAutoEncoderModel.__init__(self, img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits*2, alpha, beta, gamma)

        self.delta=delta
        self.similarity_input=Input((1,),name='similarity_input')
        self.hash_x1 = self.crop(0,0,hash_bits)
        self.hash_x2 = self.crop(0,hash_bits,hash_bits*2-1)


    def net_loss(self, y_true, y_pred):
        supervised_loss = 0  # objectives.categorical_crossentropy(y_true, y_pred)  # we calculate supervised loss outside


        self.hae.net_loss(y_true, y_pred)

        binary_loss1 = - tf.reduce_mean(tf.square(self.hash_x1 - 0.5))
        balance_loss1 = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x1, 0) - 0.5))

        binary_loss2 = - tf.reduce_mean(tf.square(self.hash_x2 - 0.5))
        balance_loss2 = tf.reduce_sum(tf.square(tf.reduce_mean(self.hash_x2, 0) - 0.5))

        decoded_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        if K.equal(self.similarity_input, 1):
            similarity_loss = K.categorical_crossentropy(self.hash_x1,self.hash_x2)
        else :
            similarity_loss = K.categorical_crossentropy(self.hash_x1,K.update_add(K.dot( self.hash_x2,-1),1 ))
        return supervised_loss + self.alpha *( binary_loss1 +binary_loss2)+ self.beta *( balance_loss1 +   balance_loss2)+ self.gamma * decoded_loss + self.delta*similarity_loss

    @staticmethod
    def crop(dimension, start, end):
        # Crops (or slices) a Tensor on a given dimension from start to end
        # example : to crop tensor x[:, :, 5:10]
        # call slice(2, 5, 10) as you want to crop on the second dimension
        def func(x):
            if dimension == 0:
                return x[start: end]
            if dimension == 1:
                return x[:, start: end]
            if dimension == 2:
                return x[:, :, start: end]
            if dimension == 3:
                return x[:, :, :, start: end]
            if dimension == 4:
                return x[:, :, :, :, start: end]
        return Lambda(func)
'''