import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Activation

isize = 512
featureDim = (isize, )
labelDim = 1000
hiddenLayDim = 2048


def build_model(batch_size=128):
    model = Sequential()
    model.add(Dense(hiddenLayDim, input_dim=isize))
    model.add(Activation('sigmoid')) #hidden layer 1
    model.add(Dense(hiddenLayDim))
    model.add(Activation('sigmoid')) #hidden layer 2
    model.add(Dense(hiddenLayDim))
    model.add(Activation('sigmoid')) #hidden layer 3
    model.add(Dense(hiddenLayDim))
    model.add(Activation('sigmoid')) #hidden layer 4
    model.add(Dense(hiddenLayDim))
    model.add(Activation('sigmoid')) #hidden layer 5
    model.add(Dense(hiddenLayDim))
    model.add(Activation('sigmoid')) #hidden layer 6
    model.add(Dense(labelDim, activation="softmax"))
    return model
