import theano.tensor as T
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer,\
    MaxPool2DLayer, ReshapeLayer

featureDim = (3, 224, 224)
labelDim = 1000


def build_model(batch_size=128):
    x = T.tensor4('input')
    layer = InputLayer((batch_size,) + featureDim, input_var=x)

    layer = Conv2DLayer(layer, 96, 11, stride=4, pad='same')
    layer = MaxPool2DLayer(layer, 3, stride=2)

    layer = Conv2DLayer(layer, 256, 5, pad='same')
    layer = MaxPool2DLayer(layer, 3, stride=2)

    layer = Conv2DLayer(layer, 384, 3, pad='same')
    layer = Conv2DLayer(layer, 384, 3, pad='same')
    layer = Conv2DLayer(layer, 256, 3, pad='same')
    layer = MaxPool2DLayer(layer, 3, stride=2)

    layer = ReshapeLayer(layer, (-1, 256 * 6 * 6))
    layer = DenseLayer(layer, 4096)
    layer = DenseLayer(layer, 4096)
    layer = DenseLayer(layer, labelDim, nonlinearity=None)

    return layer, x
