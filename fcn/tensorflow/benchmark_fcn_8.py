from builtins import range
from datetime import datetime
import numpy as np
import math
import time

import tensorflow.python.platform
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_batches', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('forward_only', False,
                            """Only run the forward pass.""")
tf.app.flags.DEFINE_boolean('forward_backward_only', False,
                            """Only run the forward-forward pass.""")

featureDim = 26752
labelDim = 26752
hiddenLayDim = 2048
numMinibatches = 100


parameters = []



def createFakeData(count):
  features = np.random.randn(count, featureDim)
  labels = np.random.randint(0, labelDim, size=(count, 1))
  return features, labels

fake_features, fake_labels = createFakeData(1024)

def getFakeMinibatch(minibatchSize):
  feat = fake_features[:minibatchSize]
  l = fake_labels[:minibatchSize]
  lab = np.zeros((minibatchSize, labelDim))
  for i in range(lab.shape[0]):
    lab[i][l[i]] = 1
  return tf.to_float(feat), tf.to_float(lab)
  #return feat, lab


# Get random parameters initialized with a iniform distribution between -0.5 and 0.5
def getParameters(name, shape):
  return tf.get_variable(name, shape, initializer=tf.random_uniform_initializer(-0.5, 0.5), dtype=tf.float32, trainable=True)

def sigmoidDNNLayer(layerIdx, input, inputDim, outputDim):
  global parameters
  W = getParameters("W" + str(layerIdx), [inputDim, outputDim])
  B = getParameters("B" + str(layerIdx), [outputDim])
  parameters += [W, B]
  return tf.nn.sigmoid(tf.nn.xw_plus_b(input, W, B))


def inference(features):
  HL0 = sigmoidDNNLayer(0, features, featureDim, hiddenLayDim)
  HL1 = sigmoidDNNLayer(1, HL0, hiddenLayDim, hiddenLayDim)
  HL2 = sigmoidDNNLayer(2, HL1, hiddenLayDim, hiddenLayDim)
  HL3 = sigmoidDNNLayer(3, HL2, hiddenLayDim, hiddenLayDim)
  HL4 = sigmoidDNNLayer(4, HL3, hiddenLayDim, hiddenLayDim)
  HL5 = sigmoidDNNLayer(5, HL4, hiddenLayDim, hiddenLayDim)

  outputLayerW = getParameters("W8", [hiddenLayDim, labelDim])
  outputLayerB = getParameters("B8", [labelDim])
  outputLayer = tf.nn.softmax(tf.nn.xw_plus_b(HL5, outputLayerW, outputLayerB))

  return outputLayer

def loss(outputLayer, labels):
  crossEntropy = -tf.reduce_mean(labels * tf.log(outputLayer))
  return crossEntropy


def time_tensorflow_run(session, target, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  if not isinstance(target, list):
    target = [target]
  target_op = tf.group(*target)
  for i in range(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target_op)
    duration = time.time() - start_time
    if i > num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))

def run_benchmark():
  global parameters
  with tf.Graph().as_default():
    features, labels = getFakeMinibatch(FLAGS.batch_size)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    last_layer = inference(features)

    # Build an initialization operation.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session('')
    sess.run(init)

    run_forward = True
    run_forward_backward = True
    if FLAGS.forward_only and FLAGS.forward_backward_only:
      raise ValueError("Cannot specify --forward_only and "
                       "--forward_backward_only at the same time.")
    if FLAGS.forward_only:
      run_forward_backward = False
    elif FLAGS.forward_backward_only:
      run_forward = False

    if run_forward:
      # Run the forward benchmark.
      time_tensorflow_run(sess, last_layer, "Forward")

    if run_forward_backward:
      # Add a simple objective so we can calculate the backward pass.
      objective = loss(last_layer, labels)
      # Compute the gradient with respect to all the parameters.
      grad = tf.gradients(objective, parameters)
      # Run the backward benchmark.
      time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
