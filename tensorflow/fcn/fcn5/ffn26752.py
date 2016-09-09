import tensorflow as tf
import numpy as np
from datetime import datetime

featureDim = 26752
labelDim = 26752
hiddenLayDim = 2048
numMinibatches = 100

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('logDevicePlacement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('noInputFeed', False,
                            """Whether to not feed new features/labels data for each minibatch.""")

def createFakeData(count):
    features = np.random.randn(count, featureDim)
    labels = np.random.randint(0, labelDim, size=(count, 1))
    return features, labels

features, labels = createFakeData(1024)

# Get random parameters initialized with a iniform distribution between -0.5 and 0.5
def getParameters(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5))

def sigmoidDNNLayer(layerIdx, input, inputDim, outputDim):
    W = getParameters("W" + str(layerIdx), [inputDim, outputDim])
    B = getParameters("B" + str(layerIdx), [outputDim])
    return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))

def getFakeMinibatch(minibatchSize):
    feat = features[:minibatchSize]
    l = labels[:minibatchSize]
    lab = np.zeros((minibatchSize, labelDim))
    for i in range(lab.shape[0]):
        lab[i][l[i]] = 1
    return feat, lab

def getLossAndAccuracyForSubBatch(features, labels):

    HL0 = sigmoidDNNLayer(0, features, featureDim, hiddenLayDim)
    HL1 = sigmoidDNNLayer(1, HL0, hiddenLayDim, hiddenLayDim)
    HL2 = sigmoidDNNLayer(2, HL1, hiddenLayDim, hiddenLayDim)

    outputLayerW = getParameters("W5", [hiddenLayDim, labelDim])
    outputLayerB = getParameters("B5", [labelDim])
    outputLayer = tf.nn.softmax(tf.nn.xw_plus_b(HL2, outputLayerW, outputLayerB))

    crossEntropy = -tf.reduce_mean(labels * tf.log(outputLayer))
    predictionCorrectness = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(predictionCorrectness, "float"))

    return crossEntropy, accuracy

def printTrainingStats(numGPUs, minibatchSize, perMinibatchTime):
    meanTimePerMinibatch = np.mean(perMinibatchTime)
    medianTimePerMinibatch = np.median(perMinibatchTime)
    minTimePerMinibatch = np.min(perMinibatchTime)

    def samplesPerSec(minibatchSize, processingTime):
        return minibatchSize/processingTime

    print('*****************************Training on %d GPUs***************************************' % numGPUs)
    print('MinibatchSize=%d, NumMinibatches=%d.' % (minibatchSize, numMinibatches))
    print('Training speed (samples/sec): Average=%d, Median=%d, Max=%d' % (samplesPerSec(minibatchSize, meanTimePerMinibatch),
                                                                           samplesPerSec(minibatchSize, medianTimePerMinibatch),
                                                                           samplesPerSec(minibatchSize, minTimePerMinibatch)))
    print('*************************************************************************************')
    print ('fake %s: fc across %d steps, %.6f sec / batch' % (datetime.now(),  numMinibatches, meanTimePerMinibatch))



