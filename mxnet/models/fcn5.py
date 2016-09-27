#!/usr/bin/env python
import mxnet as mx
import numpy as np



featureDim = (26752, )
numClasses = 26752
hiddenLayDim = 2048

def build_model():
	data = mx.sym.Variable('data')
	label = mx.sym.Variable('label')
	l1 = mx.sym.FullyConnected(data = data, num_hidden = hiddenLayDim, name = 'h1')
	a1 = mx.sym.Activation(data = l1, act_type = 'sigmoid', name = 'act1')
	l2 = mx.sym.FullyConnected(data = a1, num_hidden = hiddenLayDim, name = 'h2')
	a2 = mx.sym.Activation(data = l2, act_type = 'sigmoid', name = 'act2')
	l3 = mx.sym.FullyConnected(data = a2, num_hidden = hiddenLayDim, name = 'h3')
	a3 = mx.sym.Activation(data = l3, act_type = 'sigmoid', name = 'act3')
	w = mx.sym.Variable('weight')
	l4 = mx.sym.FullyConnected(data = a3, num_hidden = numClasses, weight = w, name = 'output')
	softmax = mx.sym.SoftmaxOutput(data = l4, label = label, name = 'softmax')
	return softmax