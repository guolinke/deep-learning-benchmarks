import numpy as np
import theano
import theano.tensor as T

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

from keras.utils import np_utils


import time
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alexnet, resnet)')
parser.add_argument('--batch_size', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--num_batches', '-n', type=int, default=100,
                    help='number of minibatches')

args = parser.parse_args()

input_generator = None

def get_output_size(batch_size):
    '''
    In most cases, the output size is (batch_size, ).
    But it will be (batch_size * seq_len,) when you use lstm
    '''
    return (batch_size, )

if args.arch == 'alexnet':
    from keras_models.alexnet import build_model, featureDim, labelDim
elif args.arch == 'resnet':
    from keras_models.resnet50 import build_model, featureDim, labelDim
elif args.arch == 'fcn5':
    from keras_models.fcn5 import build_model, featureDim, labelDim
elif args.arch == 'fcn8':
    from keras_models.fcn8 import build_model, featureDim, labelDim
elif args.arch == 'lstm32':
    from keras_models.lstm import build_model32 as build_model, featureDim32 as featureDim, \
        labelDim32 as labelDim, input_generator32 as input_generator, get_output_size32 as get_output_size
elif args.arch == 'lstm64':
    from keras_models.lstm import build_model64 as build_model, featureDim64 as featureDim, \
        labelDim64 as labelDim, input_generator64 as input_generator, get_output_size64 as get_output_size
else:
    raise ValueError('Invalid architecture name')

NUM_STEPS_BURN_IN = 10

def time_theano_run(model, inputs, labels, info_string):
    num_batches = args.num_batches
    durations = []
    for i in range(num_batches + NUM_STEPS_BURN_IN):
        start_time = time.time()
        _ = model.train_on_batch(inputs,labels)
        duration = time.time() - start_time
        if i > NUM_STEPS_BURN_IN:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - NUM_STEPS_BURN_IN, duration))
            durations.append(duration)
    durations = np.array(durations)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batches,
           durations.mean(), durations.std()))


def main():
    batch_size = args.batch_size
    print('Building model...')
    model = build_model(batch_size=batch_size)
    print("number of parameters in model: %d" % model.count_params())

    print('Compiling theano functions...')
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))
    print('Functions are compiled')

    print('input_size:', (batch_size,) + featureDim)
    num_batches = args.num_batches


    # generate input
    if input_generator is None:
        inputs = np.random.rand(batch_size, *featureDim).astype(np.float32)
    else:
        inputs = input_generator(batch_size)


    # generate label
    output_size = get_output_size(batch_size)
    
    inp_labels = np.random.randint(0, labelDim, size=output_size).astype(np.int32)

    labels = np_utils.to_categorical(inp_labels, labelDim).astype(np.float32)
    
    time_theano_run(model, inputs, labels, 'Forward-Backward')

if __name__ == '__main__':
    main()
