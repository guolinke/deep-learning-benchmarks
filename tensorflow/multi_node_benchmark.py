from datetime import datetime
import math
import time


import numpy as np
import argparse
import os
import tempfile
import tensorflow.python.platform
import tensorflow as tf

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

parser.add_argument('--worker_hosts',  default="")

parser.add_argument('--task_index', type=int, default=0)


args = parser.parse_args()



if args.arch == 'alexnet':
    from models.alexnet import build_model, featureDim, numClasses
elif args.arch == 'resnet':
    from models.resnet import build_model, featureDim, numClasses
elif args.arch == 'fcn5':
    from models.fcn5 import build_model, featureDim, numClasses
elif args.arch == 'fcn8':
    from models.fcn8 import build_model, featureDim, numClasses

used_gpus = args.gpu

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

os.environ["CUDA_VISIBLE_DEVICES"]=str(used_gpus)

worker_hosts = args.worker_hosts.split(",")
# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"worker": worker_hosts})

server = tf.train.Server(cluster,
                       job_name="worker",
                       task_index=args.task_index)

is_chief = (args.task_index == 0)

worker_device = "/job:worker/task:%d/gpu:%d" % (args.task_index, 0)

with tf.device(
        tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster)):

    batch_size = args.batch_size / len(worker_hosts)
    data_shape = (batch_size, ) + featureDim
    label_shape = (batch_size, )
    global_step = tf.Variable(0, name="global_step")

    with tf.device('/cpu:0'):
        feature = tf.Variable(np.random.uniform(0, 1, data_shape).astype(np.float32), trainable=False)
        label = tf.Variable(np.random.randint(0, numClasses, label_shape, dtype=np.int32), trainable=False)

    

    optimizer = tf.train.GradientDescentOptimizer(args.lr)
    replicas_to_aggregate = len(worker_hosts)
    global_opt = tf.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=replicas_to_aggregate,
        total_num_replicas=len(worker_hosts),
        replica_id=args.task_index,
        name="mnist_sync_replicas")

    last_layer = build_model(feature)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(last_layer, label)
    loss = tf.reduce_mean(cross_entropy)

    train_step = global_opt.minimize(loss, global_step=global_step)

    if is_chief:
        # Initial token and chief queue runners required by the sync_replicas mode
        chief_queue_runner = global_opt.get_chief_queue_runner()
        init_tokens_op = global_opt.get_init_tokens_op()

    init = tf.initialize_all_variables()

    PrintParameterCount()

    sv = None

    train_dir = tempfile.mkdtemp()

    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=train_dir,
        init_op=init, 
        recovery_wait_secs=1,
        global_step=global_step)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                           device_filters=["/job:worker/task:%d" % args.task_index])

    if is_chief:
      print("Worker %d: Initializing session..." % args.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            args.task_index)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % args.task_index)

    if is_chief:
        # Chief worker will start the chief queue runner and call the init op
        print("Starting chief queue runner and running init_tokens_op")
        sv.start_queue_runners(sess, [chief_queue_runner])
        sess.run(init_tokens_op)


    duration = time_tensorflow_run(sess, [train_step, global_step], args.num_batches, '[copy + forward + backward + update]')

    print('********************** Training on GPU('+str(args.gpu)+') **********************')
    print('Avg elasped time per mini-batch (sec/mini-batch): '+str(round(duration / args.num_batches, 6)) )
    print('Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batches)/duration))))
