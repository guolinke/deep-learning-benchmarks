import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import get_output, get_all_params
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
use_onehot_label = False

if args.arch == 'alexnet':
    from models.alexnet import build_model, featureDim, labelDim
elif args.arch == 'resnet':
    from models.resnet50 import build_model, featureDim, labelDim
elif args.arch == 'fcn5':
    from models.fcn5 import build_model, featureDim, labelDim
    use_onehot_label = True
elif args.arch == 'fcn8':
    from models.fcn8 import build_model, featureDim, labelDim
    use_onehot_label = True
elif args.arch == 'lstm32':
    from models.lstm import build_model32 as build_model, featureDim32 as featureDim, labelDim32 as labelDim, input_generator32 as input_generator
elif args.arch == 'lstm64':
    from models.lstm import build_model64 as build_model, featureDim64 as featureDim, labelDim64 as labelDim, input_generator64 as input_generator
else:
    raise ValueError('Invalid architecture name')

NUM_STEPS_BURN_IN = 10

def time_theano_run(func, fargs, info_string):
    num_batches = args.num_batches
    durations = []
    for i in range(num_batches + NUM_STEPS_BURN_IN):
        start_time = time.time()
        _ = func(*fargs[i])
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
    layer, input_var = build_model(batch_size=batch_size)
    print("number of parameters in model: %d" % lasagne.layers.count_params(layer, trainable=True))
    if use_onehot_label:
        labels_var = T.fmatrix('labels')
    else:
        labels_var = T.ivector('labels')

    output = get_output(layer)

    loss = T.nnet.categorical_crossentropy(
        T.nnet.softmax(output), labels_var).mean(
        dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.1)

    # gradient = T.grad(loss, params, disconnected_inputs="warn")
    # We add update, so we don't have to calc the gradient now.

    print('Compiling theano functions...')
    forward_func = theano.function([input_var], output)
    full_func = theano.function([input_var, labels_var], output, updates=updates)
    print('Functions are compiled')

    print('input_size:', (batch_size,) + featureDim)
    num_batches = args.num_batches


    # generate input
    if input_generator is None:
        inputs = [[np.random.rand(batch_size, *featureDim).astype(np.float32)] for i in xrange(num_batches + NUM_STEPS_BURN_IN)]
    else:
        inputs = [[input_generator(batch_size, *featureDim)] for i in xrange(num_batches + NUM_STEPS_BURN_IN)]


    # generate label
    if use_onehot_label:
        # convert the random labels to one-hot format
        labels = []
        for i in xrange(num_batches + NUM_STEPS_BURN_IN):
            batch_label = np.zeros((batch_size, labelDim)).astype(np.float32)
            for j, v in enumerate(np.random.randint(0, labelDim, size=batch_size)):
                batch_label[j][v] = 1.
            labels.append([batch_label])
    else:
        labels = [[np.random.randint(0, labelDim, size=batch_size).astype(np.int32)] for i in xrange(num_batches + NUM_STEPS_BURN_IN)]

    time_theano_run(forward_func, inputs, 'Forward')
    time_theano_run(full_func, [(ipt + lab) for ipt, lab  in zip(inputs, labels)], 'Forward-Backward')

if __name__ == '__main__':
    main()
