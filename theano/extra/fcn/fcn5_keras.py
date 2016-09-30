import os
# print(os.environ["THEANO_FLAGS"])
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import theano
import time
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from keras.utils import np_utils

nruns = 200
bsize = 64
isize = 26752
hsize = 2048
osize = 26752

#fake data
X = np.random.rand(bsize, isize).astype(np.float32)
ind = np.random.randint(0,osize,bsize)
y = np_utils.to_categorical(ind, osize).astype(np.float32)

#model definition
model = Sequential()
model.add(Dense(hsize, input_dim=isize))
model.add(Activation('sigmoid')) #hidden layer 1
model.add(Dense(hsize))
model.add(Activation('sigmoid')) #hidden layer 2
model.add(Dense(hsize))
model.add(Activation('sigmoid')) #hidden layer 3
model.add(Dense(osize))
model.add(Activation('softmax')) #output layer
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))


losses = []
#start training and measuring
start = time.time()
for i in range(nruns):
    losses.append(model.train_on_batch(X, y))
end = time.time()
print('1 GPU: {0} samples per sec'.format(nruns * bsize / (end-start)))
print('1 GPU: {0}s per batch'.format((end-start)/nruns))
for loss in losses[:10]:
     print loss
