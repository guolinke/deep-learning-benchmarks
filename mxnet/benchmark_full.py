#!/usr/bin/env python
import mxnet as mx
import numpy as np
import time
import argparse
import math


# Argument
parser=argparse.ArgumentParser()
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alexnet, resnet, fcn5, fcn8)')

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-batch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
args=parser.parse_args()


if args.arch == 'alexnet':
    from models.alexnet import build_model, featureDim, numClasses
elif args.arch == 'resnet':
    from models.resnet import build_model, featureDim, numClasses
elif args.arch == 'fcn5':
    from models.fcn5 import build_model, featureDim, numClasses
elif args.arch == 'fcn8':
    from models.fcn8 import build_model, featureDim, numClasses
else:
    raise ValueError('Invalid architecture name')

def GetTrainableParameters(symbol, executor):
    param_keys = symbol.list_arguments()
    arg = []
    grad = []
    for name in param_keys:
        if name not in ['data','label']:
            arg.append(executor.arg_dict[name])
            grad.append(executor.grad_dict[name])
    return arg, grad

def PrintShapes(symbol, executor):
    netwark_args, _ = GetTrainableParameters(symbol, executor)
    print 'Network shape checking:'
    total_para = 0
    for arg in netwark_args:
        shape = arg.shape
        print shape
        tmp_para = 1
        for dim in shape:
            tmp_para *= dim
        total_para += tmp_para
    return total_para


def TimeMxnetRun(func, iter, info = None):
    n_burn_in = 10
    for i in range(n_burn_in):
        func()
    tic = time.time()
    for i in range(iter):
        func()
    mx.nd.waitall()
    toc = time.time()
    if info:
        print ('Used time for %s : %f' %(info, (toc - tic) / iter))
    return (toc - tic)


print '************** Config Message **************'
print 'model: ' + str(args.arch)
print 'gpu: '+str(args.gpu)
print 'Mini_batch_size: '+str(args.batch_size)
print 'Num_batch: '+str(args.num_batch)
print '********************************************'


symbol = build_model()

# Basic Info
dev = mx.gpu(args.gpu)
batch_size = args.batch_size
data_shape = (batch_size, ) + featureDim
label_shape = (batch_size, )


input_shapes = {"data": data_shape, "label": label_shape}
executor = symbol.simple_bind(ctx = dev, grad_req = 'write', **input_shapes)

# Parameter counting
print 'Parameter Number: '+str(PrintShapes(symbol, executor))



# Genarate fake data
data = np.random.uniform(0, 1, data_shape).astype(np.float32)
label = np.random.randint(0, numClasses, label_shape).astype(np.int32)

# Block all async all
mx.nd.waitall()

arg, grad = GetTrainableParameters(symbol, executor)


def Full():
    executor.arg_dict['data'][:] = data
    executor.arg_dict['label'][:] = label
    executor.forward(is_train = True)
    executor.backward()
    for j in range(len(arg)):
        arg[j][:] -= args.lr * grad[j]


elasped_time = TimeMxnetRun(Full, args.num_batch, '[copy + forward + backward + update]')


print '********************** Training on GPU('+str(args.gpu)+') **********************'
print 'Avg elasped time per mini-batch (sec/mini-batch): '+str(round(elasped_time/args.num_batch, 6))
print 'Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batch)/elasped_time)))
print '****************************************************************'
