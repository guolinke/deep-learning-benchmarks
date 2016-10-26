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

parser.add_argument('--gpu', type=int, default=0)

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

device_str = '/gpu:%d'%int(args.gpu)

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


def time_tensorflow_run(session, target, num_steps, feed_dict = None, info=None):
    num_burn_in = 10
    for i in xrange(num_burn_in):
        session.run(target, feed_dict = feed_dict)
    start_time = time.time()
    for i in xrange(num_steps):
        session.run(target, feed_dict = feed_dict)
    duration = time.time() - start_time
    if info:
        print ('Used time for %s : %f' %(info, duration / num_steps))
    return duration



batch_size = args.batch_size
data_shape = (batch_size, ) + featureDim
label_shape = (batch_size, )


forward_time_data_in_gpu = 0.0

with tf.Graph().as_default(), tf.device(device_str):
    feature = tf.Variable(np.random.uniform(0, 1, data_shape).astype(np.float32), trainable=False)
    label = tf.Variable(np.random.randint(0, numClasses, label_shape, dtype=np.int32), trainable=False)

    last_layer = build_model(feature)

    init = tf.initialize_all_variables()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(init)
    
    forward_time_data_in_gpu = time_tensorflow_run(sess, tf.group(last_layer), args.num_batches)

tf.reset_default_graph()




feature_in = np.random.uniform(0, 1, data_shape).astype(np.float32)
label_in = np.random.randint(0, numClasses, label_shape, dtype=np.int32)

with tf.Graph().as_default(), tf.device(device_str):

    feature = tf.placeholder(tf.float32, data_shape)
    label = tf.placeholder(tf.int32, label_shape)
    
    last_layer = build_model(feature)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(last_layer, label)

    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(args.lr)

    grads_vars = optimizer.compute_gradients(loss)

    grad = [ x[0] for x in grads_vars ]
    train_step = optimizer.apply_gradients(grads_vars)

    init = tf.initialize_all_variables()

    PrintParameterCount()

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(init)

    forward_time_data_in_cpu = time_tensorflow_run(sess, tf.group(last_layer) , args.num_batches , {feature: feature_in, label: label_in}, '[copy + forward]')

    print ('Used time for %s : %f' %('[copy]', (forward_time_data_in_cpu - forward_time_data_in_gpu) / args.num_batches))

    time_tensorflow_run(sess, tf.group(*grad), args.num_batches, {feature: feature_in, label: label_in}, '[copy + forward + backward]')

    duration = time_tensorflow_run(sess, [train_step], args.num_batches, {feature: feature_in, label: label_in}, '[copy + forward + backward + update]')

    print('********************** Training on GPU('+str(args.gpu)+') **********************')
    print('Avg elasped time per mini-batch (sec/mini-batch): '+str(round(duration / args.num_batches, 6)) )
    print('Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batches)/duration))))
