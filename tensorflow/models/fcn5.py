import tensorflow as tf
import numpy as np

input_size = 26752
featureDim = (input_size, )
numClasses = 26752
hiddenLayDim = 2048

def getParameters(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5))

def sigmoidDNNLayer(layerIdx, input, inputDim, outputDim):
    W = getParameters("W" + str(layerIdx), [inputDim, outputDim])
    B = getParameters("B" + str(layerIdx), [outputDim])
    return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))

def build_model(features):

    HL0 = sigmoidDNNLayer(0, features, input_size, hiddenLayDim)
    HL1 = sigmoidDNNLayer(1, HL0, hiddenLayDim, hiddenLayDim)
    HL2 = sigmoidDNNLayer(2, HL1, hiddenLayDim, hiddenLayDim)

    outputLayerW = getParameters("W5", [hiddenLayDim, numClasses])
    outputLayerB = getParameters("B5", [numClasses])

    outputLayer = tf.nn.xw_plus_b(HL2, outputLayerW, outputLayerB)

    return outputLayer