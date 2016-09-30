
import theano
import theano.tensor as T
import time
import numpy as np
from collections import OrderedDict

nruns = 200
bsize = 64
isize = 26752
hsize = 2048
osize = 26752

#fake data
X = np.random.rand(bsize, isize).astype(np.float32)
y = np.zeros((bsize, osize), dtype=np.float32)
ind = np.random.randint(0, osize, bsize)
for i in range(bsize):
    y[i,ind[i]] = 1.

input_var  = T.fmatrix('input')
labels_var = T.fmatrix('labels')


def GlorotInit(rng, param_size, name=None):
    W_bound = 4 * np.sqrt(6.0 / (param_size[0] + param_size[1]))
    W = theano.shared(np.asarray(
                      rng.uniform(low=-W_bound, high=W_bound, size=param_size),
                      dtype=theano.config.floatX), borrow=True,
                      name=name)
    return W

params_size = 0

def dense_layer(ipt, input_size, output_size, name="", linear=False):
    w = theano.shared(np.random.randn(input_size, output_size).astype(np.float32) / (input_size + output_size) ** 0.5 , name="w_" + name)
    # If this GlorotInit is used, the loss will decend extreamly fast
    # w = GlorotInit(np.random, (input_size, output_size), "w_" + name)
    b = theano.shared(np.zeros((output_size,)).astype(np.float32), name="b_" + name)
    global params_size
    params_size += w.size + b.size
    layer = T.dot(ipt, w) + b.dimshuffle('x', 0)
    if not linear:
        layer = T.nnet.sigmoid(layer)
    return layer, w, b

# hidden layer 1
layer1, w1, b1 = dense_layer(input_var, isize, hsize, "1")

# hidden layer 2
layer2, w2, b2 = dense_layer(layer1, hsize, hsize, "2")

# hidden layer 3
layer3, w3, b3 = dense_layer(layer2, hsize, hsize, "3")

# output layer
output, w4, b4 = dense_layer(layer3, hsize, osize, "4", linear=True)


loss = T.nnet.categorical_crossentropy(
    T.nnet.softmax(output), labels_var).mean(
    dtype=theano.config.floatX)


# calc the learning rate and define updates
learning_rate = 0.1

params = [
    w1, b1,
    w2, b2,
    w3, b3,
    w4, b4
]
'''
params = lasagne.layers.get_all_params(layer, trainable=True)
'''

grads = T.grad(cost=loss, wrt=params)
updates = OrderedDict()
for p, g in zip(params, grads):
    updates[p] = p - g * learning_rate

# theano.printing.debugprint(loss)
full_func = theano.function([input_var, labels_var], loss, updates=updates)


losses = []
start = time.time()
for i in range(nruns):
    losses.append(full_func(X, y))
end = time.time()

print "Parameter number:", params_size.eval()
print '1 GPU: {0} samples per sec'.format(nruns * bsize / (end-start))
print '1 GPU: {0}s per batch'.format((end-start)/nruns)
print 'Losses:'
for loss in losses[:10]:
     print loss
