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

def LogLoss(preb, label):
    s = 0
    for i in range(len(label)):
        s -= math.log(preb[i][label[i]])
    return s/len(label)

def TimeMxnetRun(symbol, executor, iter, data, label, lr):
    arg, grad = GetTrainableParameters(symbol, executor)
    kBurnIn = 20
    for i in range(kBurnIn):
        executor.arg_dict['data'][:] = data
        executor.arg_dict['label'][:] = label
        executor.forward(is_train = True)
        executor.backward()
        for i in range(len(arg)):
            arg[i][:] -= lr * grad[i]
    mx.nd.waitall()
    tic = time.time()
    for i in range(iter):
        executor.arg_dict['data'][:] = data
        executor.arg_dict['label'][:] = label
        executor.forward(is_train = True)
        #print LogLoss(executor.outputs[0].asnumpy(), label)
        executor.backward()
        for i in range(len(arg)):
            arg[i][:] -= lr * grad[i]
    mx.nd.waitall()
    toc = time.time()
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

# Init
for arg in executor.arg_arrays:
    arg[:] = mx.rnd.uniform(-0.01, 0.01, arg.shape)

# Genarate fake data
data = np.random.uniform(-1, 1, data_shape).astype("float32")
label = np.random.randint(0, numClasses, label_shape)

# Block all async all
mx.nd.waitall()

# Test
elasped_time = float(TimeMxnetRun(symbol, executor, args.num_batch, data, label, args.lr))

print '********************** Training on GPU('+str(args.gpu)+') **********************'
print 'Avg elasped time per mini-batch (sec/mini-batch): '+str(round(elasped_time/args.num_batch, 6))
print 'Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batch)/elasped_time)))
print '****************************************************************'