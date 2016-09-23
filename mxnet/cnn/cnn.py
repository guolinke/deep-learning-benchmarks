#!/usr/bin/env python
import mxnet as mx
import numpy as np
import time
import argparse
from symbol_alexnet import get_alexnet_symbol
from symbol_resnet import get_resnet_symbol

def param_count(arg_names, arg_map):
    print args.network + ' shape checking:'
    total_para = 0
    for i in range(len(arg_names)):
        if i == 0:
            continue
        shape = arg_map[arg_names[i]].asnumpy().shape
        print shape
        tmp_para = 1
        for dim in shape:
            tmp_para *= dim
        total_para += tmp_para
        #print 'tmp para:'+str(tmp_para)
    return total_para

def test_full(model, mb, param_blocks, grad):
    param_len = len(param_blocks)
    tic = time.time()
    for i in range(mb):
        model.forward(is_train = True)
        model.backward([grad])
        for i in range(param_len):
            param_blocks[i][1][:] -= 0.001 * param_blocks[i][2][:]
    mx.nd.waitall()
    toc = time.time()
    return (toc - tic)

# Argument
parser=argparse.ArgumentParser(description='choose model to test benchmark')
parser.add_argument('--network', type=str, default='alexnet')
parser.add_argument('--pad', type=str, default='pad', choices=['pad', 'discard', 'roll_over'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-batch', type=int, default=100)
parser.add_argument('--printmsg', type=bool, default=True)
args=parser.parse_args()

# Dump out msg
if args.printmsg == True:
    print '************** Config Message **************'
    print 'network: '+str(args.network.upper())
    print 'gpu: '+str(args.gpu)
    print 'last_batch_handle: '+str(args.pad)
    print 'Mini_batch_size: '+str(args.batch_size)
    print 'Num_batch: '+str(args.num_batch)
    print '********************************************'

# Basic Info
dev = mx.gpu(args.gpu)
batch_size = args.batch_size
dshape = (batch_size, 3, 224, 224)
#lshape = (batch_size)

# Mock data iterator
tmp_data = np.random.uniform(-1, 1, dshape).astype("float32")
train_iter = mx.io.NDArrayIter(data=tmp_data, batch_size=batch_size, shuffle=False, last_batch_handle=args.pad)

# Bind to get executor
# This is what happened behind mx.model.Feedforward
if args.network == 'alexnet':
    net = get_alexnet_symbol()
elif args.network == 'resnet':
    net = get_resnet_symbol()
net_exec = net.simple_bind(ctx=dev, grad_req='write', data=dshape)
#print 'net is okay'

# Data structure
arg_names = net.list_arguments()
arg_map = dict(zip(arg_names, net_exec.arg_arrays))
grad_map = dict(zip(arg_names, net_exec.grad_arrays))
param_blocks = [(i, arg_map[arg_names[i]], grad_map[arg_names[i]]) for i in range(len(arg_names)) if grad_map[arg_names[i]] != None]
grad = mx.nd.zeros((batch_size, 1000), ctx=dev)

# Init
for i in range(len(param_blocks)):
    param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)
    param_blocks[i][2][:] = 0.

# Set data
train_iter.reset()
dbatch = train_iter.next()
dbatch.data[0].copyto(arg_map['data'])
# block all async all
mx.nd.waitall()

# Parameter counting
print 'Parameter Number: '+str(param_count(arg_names, arg_map))

# Test
elasped_time = float(test_full(net_exec, args.num_batch, param_blocks, grad))
print '********************** Training on GPU('+str(args.gpu)+') **********************'
print 'Avg elasped time per mini-batch (sec/mini-batch): '+str(round(elasped_time/args.num_batch, 6))
print 'Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batch)/elasped_time)))
print '****************************************************************'