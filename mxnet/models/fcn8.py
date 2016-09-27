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
	l4 = mx.sym.FullyConnected(data = a3, num_hidden = hiddenLayDim, name = 'h4')
	a4 = mx.sym.Activation(data = l4, act_type = 'sigmoid', name = 'act4')
	l5 = mx.sym.FullyConnected(data = a4, num_hidden = hiddenLayDim, name = 'h5')
	a5 = mx.sym.Activation(data = l5, act_type = 'sigmoid', name = 'act5')
	l6 = mx.sym.FullyConnected(data = a5, num_hidden = hiddenLayDim, name = 'h6')
	a6 = mx.sym.Activation(data = l6, act_type = 'sigmoid', name = 'act6')
	w = mx.sym.Variable('weight')
	l7 = mx.sym.FullyConnected(data = a6, num_hidden = numClasses, weight = w, name = 'output')
	softmax = mx.sym.SoftmaxOutput(data = l7, label = label, name = 'softmax')
	return softmax