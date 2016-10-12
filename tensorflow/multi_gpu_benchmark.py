from datetime import datetime
import math
import time


import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--arch', '-a', default='alexnet',
                                        help='Convnet architecture \
                                        (alexnet, resnet, fcn5, fcn8)')

parser.add_argument('--batch-size', '-b', type=int, default=128,
                                        help='minibatch size')

parser.add_argument('--num-batches', '-n', type=int, default=100,
                                        help='number of minibatches')

parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument('--gpu',  default="0")

args = parser.parse_args()

import tensorflow.python.platform
import tensorflow as tf

if args.arch == 'alexnet':
    from models.alexnet import build_model, featureDim, numClasses
elif args.arch == 'resnet':
    from models.resnet import build_model, featureDim, numClasses
elif args.arch == 'fcn5':
    from models.fcn5 import build_model, featureDim, numClasses
elif args.arch == 'fcn8':
    from models.fcn8 import build_model, featureDim, numClasses

used_gpus = [int(x) for x in args.gpu.split(',')]

def PrintParameterCount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print "Parameter Number:" + str(total_parameters)

def aggregateGradients(subMinibatchGradients):
    aggGrads = []
    for gradAndVars in zip(*subMinibatchGradients):
        # Note that each gradAndVars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in gradAndVars:
            # Add 0 dimension to the gradients to represent the replica.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'replica' dimension which we will sum over below.
            grads.append(expanded_g)

        # Sum over the 'replica' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across replicas. So .. we will just return the first replica's pointer to
        # the Variable.
        v = gradAndVars[0][1]
        gradAndVar = (grad, v)
        aggGrads.append(gradAndVar)
    return aggGrads


def time_tensorflow_run(session, target, num_steps, info=None):
    num_burn_in = 10
    for i in xrange(num_burn_in):
        session.run(target)
    start_time = time.time()
    for i in xrange(num_steps):
        session.run(target)
    duration = time.time() - start_time
    if info:
        print ('Used time for %s : %f' %(info, duration / num_steps))
    return duration

batch_size = args.batch_size / len(used_gpus)
data_shape = (batch_size, ) + featureDim
label_shape = (batch_size, )

with tf.device('/cpu:0'):
    feature = tf.Variable(np.random.uniform(0, 1, data_shape).astype(np.float32), trainable=False)
    label = tf.Variable(np.random.randint(0, numClasses, label_shape, dtype=np.int32), trainable=False)


optimizer = tf.train.GradientDescentOptimizer(args.lr)
subMinibatchGradients = []
for i in used_gpus:
    with  tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ("replica", i)) as scope:
            last_layer = build_model(feature)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(last_layer, label)
            loss = tf.reduce_mean(cross_entropy)

            tf.get_variable_scope().reuse_variables()

            grads = optimizer.compute_gradients(loss)

            subMinibatchGradients.append(grads)

grads = aggregateGradients(subMinibatchGradients)

train_step = optimizer.apply_gradients(grads)

init = tf.initialize_all_variables()

PrintParameterCount()

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
sess.run(init)

duration = time_tensorflow_run(sess, [train_step], args.num_batches, '[copy + forward + backward + update]')

print('********************** Training on GPU('+str(args.gpu)+') **********************')
print('Avg elasped time per mini-batch (sec/mini-batch): '+str(round(duration / args.num_batches, 6)) )
print('Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batches)/duration))))
