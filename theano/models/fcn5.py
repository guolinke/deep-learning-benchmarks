import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer


featureDim = (26752, )
labelDim = 26752
hiddenLayDim = 2048


def build_model(batch_size=128):
    x = T.tensor3('input')
    layer = InputLayer((batch_size, 3,) + featureDim, input_var=x)
    layer = DenseLayer(layer, hiddenLayDim, nonlinearity=lasagne.nonlinearities.sigmoid)
    layer = DenseLayer(layer, hiddenLayDim, nonlinearity=lasagne.nonlinearities.sigmoid)
    layer = DenseLayer(layer, hiddenLayDim, nonlinearity=lasagne.nonlinearities.sigmoid)
    layer = DenseLayer(layer, labelDim, nonlinearity=lasagne.nonlinearities.softmax)
    return layer, x
