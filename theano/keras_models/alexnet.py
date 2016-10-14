import theano.tensor as T

from keras.layers.convolutional import Convolution2D, ZeroPadding2D

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape, Flatten
from keras.layers.pooling import MaxPooling2D

featureDim = (3, 224, 224)
labelDim = 1000


def build_model(batch_size=128):

    model = Sequential()
    model.add(ZeroPadding2D(padding=(2, 2), input_shape=featureDim) )
    model.add(Convolution2D(96, 11, 11, subsample=(4, 4), border_mode='same', activation="relu") )
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), border_mode='same'))

    model.add(Convolution2D(256, 5, 5, border_mode='same', activation="relu") )
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), border_mode='same'))

    model.add(Convolution2D(384, 3, 3, border_mode='same', activation="relu") )
    model.add(Convolution2D(384, 3, 3, border_mode='same', activation="relu") )
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation="relu") )

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), border_mode='same'))
    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(labelDim, activation="softmax"))

    return model
