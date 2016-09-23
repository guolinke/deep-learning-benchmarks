#!/usr/bin/env python
import mxnet as mx
import numpy as np
import time
import argparse

def param_count(arg_names, arg_map):
    print 'FCN-8 shape checking:'
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
parser=argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-batch', type=int, default=100)
parser.add_argument('--printmsg', type=bool, default=True)
args=parser.parse_args()

# Dump out msg
if args.printmsg == True:
    print '************** Config Message **************'
    print 'model: FCN-8'
    print 'gpu: '+str(args.gpu)
    print 'Mini_batch_size: '+str(args.batch_size)
    print 'Num_batch: '+str(args.num_batch)
    print '********************************************'

# Basic Info
dev = mx.gpu(args.gpu)
batch_size = args.batch_size
dshape = (batch_size, 26752)
#lshape = (batch_size, )

# Mock data iterator
tmp_data = np.random.uniform(-1, 1, dshape).astype("float32")
train_iter = mx.io.NDArrayIter(data=tmp_data, batch_size=batch_size, shuffle=False)
#tmp_label = np.random.uniform(-1, 1, lshape).astype("float32")

# FCN-5 network
data = mx.sym.Variable('data')
label = mx.sym.Variable('softmax_label')
l1 = mx.sym.FullyConnected(data = data, num_hidden = 2048, name = 'h1')
a1 = mx.sym.Activation(data = l1, act_type = 'sigmoid', name = 'act1')
l2 = mx.sym.FullyConnected(data = a1, num_hidden = 2048, name = 'h2')
a2 = mx.sym.Activation(data = l2, act_type = 'sigmoid', name = 'act2')
l3 = mx.sym.FullyConnected(data = a2, num_hidden = 2048, name = 'h3')
a3 = mx.sym.Activation(data = l3, act_type = 'sigmoid', name = 'act3')
l4 = mx.sym.FullyConnected(data = a3, num_hidden = 2048, name = 'h4')
a4 = mx.sym.Activation(data = l4, act_type = 'sigmoid', name = 'act4')
l5 = mx.sym.FullyConnected(data = a4, num_hidden = 2048, name = 'h5')
a5 = mx.sym.Activation(data = l5, act_type = 'sigmoid', name = 'act5')
l6 = mx.sym.FullyConnected(data = a5, num_hidden = 2048, name = 'h6')
a6 = mx.sym.Activation(data = l6, act_type = 'sigmoid', name = 'act6')
w = mx.sym.Variable('weight')
l7 = mx.sym.FullyConnected(data = a6, num_hidden = 26752, weight = w, name = 'output')
#cost_classification = mx.sym.SoftmaxOutput(data = l4, label = label)
net = l7.simple_bind(ctx = dev, grad_req = 'write', data = dshape)
#print 'net is okay'

# Data structure
arg_names = l7.list_arguments()
arg_map = dict(zip(arg_names, net.arg_arrays))
grad_map = dict(zip(arg_names, net.grad_arrays))
param_blocks = [(i, arg_map[arg_names[i]], grad_map[arg_names[i]]) for i in range(len(arg_names)) if grad_map[arg_names[i]] != None]
grad = mx.nd.zeros((batch_size, 26752), ctx = dev)

# Init
for i in range(len(param_blocks)):
    param_blocks[i][1][:] = mx.rnd.uniform(-0.01, 0.01, param_blocks[i][1].shape)
    param_blocks[i][2][:] = 0.

# Set data
train_iter.reset()
dbatch = train_iter.next()
dbatch.data[0].copyto(arg_map['data'])
# Block all async all
mx.nd.waitall()

# Parameter counting
print 'Parameter Number: '+str(param_count(arg_names, arg_map))

# Test
elasped_time = float(test_full(net, args.num_batch, param_blocks, grad))
print '********************** Training on GPU('+str(args.gpu)+') **********************'
print 'Avg elasped time per mini-batch (sec/mini-batch): '+str(round(elasped_time/args.num_batch, 6))
print 'Avg samples per second (samples/sec): '+str(int(round((args.batch_size * args.num_batch)/elasped_time)))
print '****************************************************************'